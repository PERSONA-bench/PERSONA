import json
import os
import sys
from datetime import datetime, timezone
from copy import deepcopy
from collections import deque

# --- Configuration ---
INPUT_JSON_FILE = "merged_posts_remapped.json"
# Output filename for Task: Predict Reply Body with pseudo-random context
WITH_PSEUDO_RANDOM_CONVO_OUTPUT_JSONL = "WithPseudoRandomConversationPrompts_BodyPrediction.jsonl"  # Changed output filename

skipped_post_count = 0
REMOVED_BODY_MARKER = "[removed]"
DELETED_BODY_MARKER = "[deleted]"
CONTEXT_SHIFT_OFFSET = 200  # Number of posts to shift for context


# --- Helper Functions ---

def clean_text(text):
    if text is None:
        return ""
    return str(text).replace("\n", " ").replace("\r", " ").strip()


def parse_timestamp(ts_str):
    if not ts_str:
        return None
    try:
        if isinstance(ts_str, datetime):
            return ts_str.astimezone(timezone.utc) if ts_str.tzinfo else ts_str.replace(tzinfo=timezone.utc)
        original_ts_str = ts_str
        if ts_str.endswith('Z'):
            ts_str = ts_str[:-1] + '+00:00'
        # Handle potential milliseconds or microseconds
        if '.' in ts_str and '+' in ts_str:  # Check if there's a fraction and a timezone
            ts_parts = ts_str.split('.')
            time_part_before_fraction = ts_parts[0]
            fraction_and_tz = ts_parts[1]
            tz_part_index = fraction_and_tz.find('+')  # or '-'
            if tz_part_index == -1: tz_part_index = fraction_and_tz.find('-')

            if tz_part_index != -1:
                # Truncate or round microseconds/milliseconds to what fromisoformat can handle (up to 6 digits for microseconds)
                fraction_part = fraction_and_tz[:tz_part_index]
                if len(fraction_part) > 6:
                    fraction_part = fraction_part[:6]
                ts_str = f"{time_part_before_fraction}.{fraction_part}{fraction_and_tz[tz_part_index:]}"
            else:  # No explicit timezone offset after fraction, assume Z if original ended with Z
                if original_ts_str.endswith('Z'):
                    fraction_part = fraction_and_tz
                    if len(fraction_part) > 6: fraction_part = fraction_part[:6]
                    ts_str = f"{time_part_before_fraction}.{fraction_part}+00:00"

        dt_obj = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        return dt_obj.astimezone(timezone.utc) if dt_obj.tzinfo else dt_obj.replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            dt_obj = datetime.strptime(original_ts_str, '%Y-%m-%d %H:%M:%S')
            return dt_obj.replace(tzinfo=timezone.utc)
        except ValueError:
            try:
                dt_obj = datetime.strptime(original_ts_str, '%Y-%m-%dT%H:%M:%S')
                return dt_obj.replace(tzinfo=timezone.utc)
            except ValueError:
                # print(f"Warning: Could not parse timestamp '{original_ts_str}'", file=sys.stderr)
                return None


def datetime_to_string(dt_obj):
    if isinstance(dt_obj, datetime):
        if dt_obj.tzinfo is None or dt_obj.tzinfo.utcoffset(dt_obj) is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        else:
            dt_obj = dt_obj.astimezone(timezone.utc)
        return dt_obj.isoformat().replace('+00:00', 'Z')
    return dt_obj


def convert_datetimes_to_strings_in_obj(obj, for_history=False):
    if isinstance(obj, list):
        return [convert_datetimes_to_strings_in_obj(item, for_history) for item in obj]
    elif isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if for_history and k == 'comments':
                continue
            if isinstance(v, datetime):
                new_dict[k] = datetime_to_string(v)
            elif k.startswith('_'):
                if not for_history:
                    if not k.startswith('__'):
                        continue
                elif for_history:
                    if k in ['_parsed_timestamp', '_abs_score', '_parent_id', '_parent_body', '_parent_author']:
                        continue
                    if not k.startswith('__'):
                        continue
            new_dict[k] = convert_datetimes_to_strings_in_obj(v, for_history)
        return new_dict
    return obj


def generate_unique_ids_for_post(post_data, post_index_prefix):
    flat_nodes = []
    post_id_str = str(post_index_prefix)
    post_data['id'] = post_id_str
    post_data['type'] = 'post'
    for key in ['title', 'url', 'content', '__sub__', 'score', 'author', 'timestamp']:
        post_data.setdefault(key, None)
        if key == 'content' and post_data.get(key) is None:
            post_data[key] = post_data.get('body')

    if isinstance(post_data.get('timestamp'), datetime):
        post_data['timestamp'] = datetime_to_string(post_data['timestamp'])
    elif post_data.get('timestamp') is None and 'created_utc' in post_data:  # Fallback for posts
        parsed_fallback_ts = parse_timestamp(post_data['created_utc'])
        if parsed_fallback_ts:
            post_data['timestamp'] = datetime_to_string(parsed_fallback_ts)

    flat_nodes.append(post_data)
    comment_counter = 1

    def _assign_ids_recursive(comments_list, parent_id_str_rec):
        nonlocal comment_counter
        if not isinstance(comments_list, list): return
        for comment in comments_list:
            if not isinstance(comment, dict): continue
            current_id_str = f"{parent_id_str_rec}-c{comment_counter}"
            comment_counter += 1
            comment['id'] = current_id_str
            comment['parent_id'] = parent_id_str_rec
            comment['type'] = 'comment'
            for key in ['body', 'author', 'score', 'timestamp']:
                comment.setdefault(key, None)
            if isinstance(comment.get('timestamp'), datetime):
                comment['timestamp'] = datetime_to_string(comment['timestamp'])
            elif comment.get('timestamp') is None and 'created_utc' in comment:  # Fallback for comments
                parsed_fallback_ts = parse_timestamp(comment['created_utc'])
                if parsed_fallback_ts:
                    comment['timestamp'] = datetime_to_string(parsed_fallback_ts)

            flat_nodes.append(comment)
            if 'replies' in comment and isinstance(comment['replies'], list):
                _assign_ids_recursive(comment['replies'], current_id_str)

    _assign_ids_recursive(post_data.get('comments', []), post_id_str)
    return flat_nodes


def find_author_replies_for_body_prediction(flat_nodes_list, post_author_name):
    author_replies = []
    nodes_by_id = {node['id']: node for node in flat_nodes_list}
    for node in flat_nodes_list:
        if node.get('type') == 'comment' and node.get('author') == post_author_name:
            body_text = node.get('body')
            if body_text is None:
                continue

            parent_id = node.get('parent_id')
            parent_node = nodes_by_id.get(parent_id)
            if not parent_node:  # Skip if parent not found (should not happen with good data)
                continue
            parent_body = parent_node.get('content', parent_node.get('body', ""))
            parent_author = parent_node.get('author', "")

            parsed_ts = parse_timestamp(node.get('timestamp'))
            if parsed_ts:
                reply_details = deepcopy(node)
                reply_details['_parsed_timestamp'] = parsed_ts
                reply_details['_parent_id'] = parent_id
                reply_details['_parent_body'] = clean_text(parent_body)
                reply_details['_parent_author'] = parent_author
                reply_details['_parent_score'] = parent_node.get('score')  # Store parent score
                author_replies.append(reply_details)

    author_replies.sort(key=lambda r: r['_parsed_timestamp'], reverse=True)
    return author_replies


def count_qualifying_dialogue_trees(all_nodes_in_post, post_author_name):
    if not all_nodes_in_post: return 0
    post_node = all_nodes_in_post[0]
    if post_node.get('type') != 'post': return 0  # Should be the post itself
    nodes_map = {node['id']: node for node in all_nodes_in_post}
    top_level_comment_ids = [node['id'] for node in all_nodes_in_post
                             if node.get('type') == 'comment' and node.get('parent_id') == post_node['id']]
    tree_count = 0
    for tlc_id in top_level_comment_ids:
        author_participated_in_thread = False
        # q_thread_nodes = deque() # Not strictly needed here for this check
        tlc_node_obj = nodes_map.get(tlc_id)
        if not tlc_node_obj: continue
        # q_thread_nodes.append(tlc_node_obj) # Not strictly needed here
        # visited_in_thread_check = {tlc_id} # Not strictly needed here
        temp_q = deque([tlc_node_obj])
        visited_temp = {tlc_id}
        while temp_q:
            curr_temp = temp_q.popleft()
            if curr_temp.get('author') == post_author_name:
                author_participated_in_thread = True
                break
            for node_in_map in all_nodes_in_post:  # More direct way to find children
                if node_in_map.get('parent_id') == curr_temp.get('id') and node_in_map.get('id') not in visited_temp:
                    reply_obj = nodes_map.get(
                        node_in_map.get('id'))  # Should always exist as it's from nodes_map's source
                    if reply_obj: temp_q.append(reply_obj)
                    visited_temp.add(node_in_map.get('id'))

        if author_participated_in_thread:
            # Check if TLC has any replies at all (not just author replies)
            has_replies_to_tlc = any(node.get('parent_id') == tlc_id for node in nodes_map.values())
            if has_replies_to_tlc: tree_count += 1
    return tree_count


def get_tlc_id_for_target_reply(target_reply_node, nodes_map, root_post_id):
    if not target_reply_node or target_reply_node.get('id') == root_post_id or target_reply_node.get('type') == 'post':
        return None
    current_id = target_reply_node.get('id')
    current_node = target_reply_node
    while current_node:
        parent_id = current_node.get('parent_id')
        if parent_id == root_post_id:
            return current_id  # This is the ID of the top-level comment
        if not parent_id:  # Should not happen if data is consistent
            return None
        current_node = nodes_map.get(parent_id)
        if not current_node:  # Parent not in map, error or reached end of known data
            return None
        current_id = current_node.get('id')  # Update current_id to the parent's ID for the next iteration
    return None  # Should be caught by parent_id == root_post_id


def get_nodes_in_dialogue_tree(all_post_nodes_flat, tlc_id_for_tree):
    tree_nodes_list = []
    if not tlc_id_for_tree:
        return tree_nodes_list
    nodes_to_visit_q = deque()
    tlc_node_obj = next(
        (n for n in all_post_nodes_flat if n.get('id') == tlc_id_for_tree and n.get('type') == 'comment'), None)

    if tlc_node_obj:
        nodes_to_visit_q.append(tlc_node_obj)
        tree_nodes_list.append(tlc_node_obj)  # Add TLC itself
        visited_node_ids = {tlc_id_for_tree}
        while nodes_to_visit_q:
            current_parent_node_tree = nodes_to_visit_q.popleft()
            current_parent_id_tree = current_parent_node_tree.get('id')
            for child_node in all_post_nodes_flat:  # Iterate through all nodes to find children
                if child_node.get('parent_id') == current_parent_id_tree and child_node.get('type') == 'comment':
                    child_id = child_node.get('id')
                    if child_id not in visited_node_ids:
                        tree_nodes_list.append(child_node)
                        nodes_to_visit_q.append(child_node)
                        visited_node_ids.add(child_id)
    return tree_nodes_list


def calculate_removed_percentage_in_tree(tree_nodes_list):
    if not tree_nodes_list:
        return 0.0, 0
    removed_count = 0
    actual_comment_nodes_in_tree = [node for node in tree_nodes_list if node.get('type') == 'comment']
    total_comments_in_tree = len(actual_comment_nodes_in_tree)
    if total_comments_in_tree == 0:
        return 0.0, 0
    for node_in_tree in actual_comment_nodes_in_tree:
        body_text = clean_text(node_in_tree.get('body', ""))
        if body_text == REMOVED_BODY_MARKER or body_text == DELETED_BODY_MARKER:
            removed_count += 1
    return float(removed_count) / total_comments_in_tree, total_comments_in_tree


def build_post_context_excluding_node_id(post_data_orig_hierarchical, node_id_to_remove):
    if not node_id_to_remove: return deepcopy(post_data_orig_hierarchical)

    context_data = deepcopy(post_data_orig_hierarchical)

    # Check if the post itself is the node to remove (should not happen for replies)
    if context_data.get('id') == node_id_to_remove and context_data.get('type') == 'post':
        # This case implies we are trying to remove the root post, which is not typical
        # for reply prediction context. For safety, return None or an empty structure.
        return None  # Or handle as an error / return an empty post structure

    # Recursively remove the node from comments and their replies
    def _recursive_remove(comments_list):
        if not isinstance(comments_list, list):
            return comments_list

        new_list = []
        for item in comments_list:
            if not isinstance(item, dict):
                new_list.append(item)  # Keep non-dict items if any (though not expected)
                continue

            if item.get('id') == node_id_to_remove:
                continue  # Skip this node

            # Recursively process replies of the current item
            if 'replies' in item and isinstance(item.get('replies'), list):
                item['replies'] = _recursive_remove(item['replies'])

            new_list.append(item)
        return new_list

    if 'comments' in context_data and isinstance(context_data.get('comments'), list):
        context_data['comments'] = _recursive_remove(context_data['comments'])

    return context_data


def get_truncated_dialogue_tree_for_body_fewshot(full_post_flat_nodes_fs, few_shot_reply_obj_fs, post_author_name_fs):
    if not few_shot_reply_obj_fs or not few_shot_reply_obj_fs.get('id'): return None, None, None

    nodes_map_fs = {node['id']: node for node in full_post_flat_nodes_fs}
    few_shot_reply_id_to_exclude = few_shot_reply_obj_fs['id']
    root_post_id_fs = full_post_flat_nodes_fs[0]['id']  # Assuming first node is the post

    # Determine the top-level comment (TLC) ID for the few-shot reply's thread
    tlc_id_fs = None
    current_node_id_trace = few_shot_reply_id_to_exclude
    path_to_tlc_ids = [few_shot_reply_id_to_exclude]  # Path from excluded reply up to (but not including) TLC's parent

    while current_node_id_trace:
        node_trace = nodes_map_fs.get(current_node_id_trace)
        if not node_trace or node_trace.get('type') == 'post':  # Should not happen if reply is valid
            break

        parent_id_trace = node_trace.get('parent_id')
        if not parent_id_trace:  # Should not happen
            break

        if parent_id_trace == root_post_id_fs:  # current_node_id_trace is the TLC
            tlc_id_fs = current_node_id_trace
            break

        path_to_tlc_ids.append(parent_id_trace)  # Add parent to path
        current_node_id_trace = parent_id_trace  # Move up

    if not tlc_id_fs: return None, None, None
    path_to_tlc_ids.reverse()  # Now path goes from TLC down to the parent of excluded reply

    # Function to recursively build the display tree, stopping before the few-shot reply
    def build_display_branch(current_node_id_build, stop_at_child_id, current_path_to_excluded_child):
        original_node_build = nodes_map_fs.get(current_node_id_build)
        if not original_node_build: return None

        node_copy_display_build = deepcopy(original_node_build)
        # Remove fields not needed for display or that might leak info
        node_copy_display_build.pop('replies', None)
        node_copy_display_build.pop('_parsed_timestamp', None)
        node_copy_display_build.pop('_parent_id', None)  # Internal helper fields
        node_copy_display_build.pop('_parent_body', None)
        node_copy_display_build.pop('_parent_author', None)
        node_copy_display_build.pop('_parent_score', None)

        node_copy_display_build['replies'] = []

        # Find children of original_node_build from flat list
        children_original_build = [n for n in full_post_flat_nodes_fs if n.get('parent_id') == current_node_id_build]
        # Sort children by timestamp (ensure timestamps are parsed for sorting)
        for child in children_original_build:  # Ensure _parsed_timestamp exists for sorting
            if '_parsed_timestamp' not in child:
                child['_parsed_timestamp'] = parse_timestamp(child.get('timestamp'))

        children_original_build.sort(
            key=lambda r: r.get('_parsed_timestamp') or datetime.min.replace(tzinfo=timezone.utc))

        for child_node_orig_build in children_original_build:
            if stop_at_child_id and child_node_orig_build['id'] == stop_at_child_id:
                break  # Stop at the branch leading to the actual few-shot reply

            # Determine if this child is on the direct path to the excluded few-shot reply
            is_on_direct_path = False
            if stop_at_child_id and current_path_to_excluded_child:
                try:
                    # Check if current_node_id_build is in path and child_node_orig_build is the next step
                    idx_current_in_path = current_path_to_excluded_child.index(current_node_id_build)
                    if idx_current_in_path + 1 < len(current_path_to_excluded_child) and \
                            current_path_to_excluded_child[idx_current_in_path + 1] == child_node_orig_build['id']:
                        is_on_direct_path = True
                except ValueError:  # current_node_id_build not in path (shouldn't happen if logic is correct)
                    pass

            # Recurse: if on direct path, pass the stop_at_child_id down. Otherwise, don't.
            branch_for_child_build = build_display_branch(
                child_node_orig_build['id'],
                stop_at_child_id if is_on_direct_path else None,
                current_path_to_excluded_child if is_on_direct_path else []
            )
            if branch_for_child_build:
                node_copy_display_build['replies'].append(branch_for_child_build)

        return node_copy_display_build

    # Build the truncated tree starting from the TLC of the few-shot reply
    # The `path_to_tlc_ids` here is actually the path from TLC TO the parent of the excluded comment.
    # The `few_shot_reply_id_to_exclude` is the comment whose branch we truncate.
    truncated_tlc_tree_fs = build_display_branch(tlc_id_fs, few_shot_reply_id_to_exclude, path_to_tlc_ids)

    if not truncated_tlc_tree_fs: return None, None, None

    # Prepare parent info for the few-shot prompt
    parent_of_few_shot_node = nodes_map_fs.get(few_shot_reply_obj_fs.get('_parent_id'))
    parent_of_few_shot_info_str = "a comment"  # Default
    if parent_of_few_shot_node:
        parent_author_fs = parent_of_few_shot_node.get('author', 'Unknown User')
        parent_body_fs = clean_text(parent_of_few_shot_node.get('body', ''))
        parent_id_fs = parent_of_few_shot_node.get('id', 'unknown_id')
        parent_score_fs = parent_of_few_shot_node.get('score', 'N/A')
        parent_of_few_shot_info_str = (f"a comment by '{parent_author_fs}' "
                                       f"(ID: {parent_id_fs}, Score: {parent_score_fs}) "
                                       f"which says: \"{parent_body_fs}\"")

    # Convert the structured tree to JSON string for the prompt
    # The structure is a single TLC, so wrap it in a list for consistency if needed by prompt format
    few_shot_context_json_str = json.dumps([convert_datetimes_to_strings_in_obj(truncated_tlc_tree_fs)], indent=2)

    # Create the few-shot prompt segment
    fs_prompt_segment = (
        "### FEW-SHOT EXAMPLE ###\n"
        "Consider the following dialogue tree context from a previous post (the author's target reply is missing, and any subsequent replies in that specific branch are truncated):\n"
        f"{few_shot_context_json_str}\n"
        f"The author of that previous post, '{post_author_name_fs}', made a reply to {parent_of_few_shot_info_str}. "
        f"(This reply by '{post_author_name_fs}' had ID: {few_shot_reply_obj_fs['id']}).\n"
        "The actual reply body from '{post_author_name_fs}' was:"
    )
    fs_true_answer = clean_text(few_shot_reply_obj_fs.get('body'))

    return fs_prompt_segment, fs_true_answer


def get_default_few_shot_for_body_prediction():
    default_dialogue_tree_context = [{
        "id": "default_post1-c1", "type": "comment", "parent_id": "default_post1",
        "author": "UserX",
        "body": "This is a great post about topic Y. I have a follow-up question regarding its implications for Z.",
        "timestamp": "2024-01-02T10:00:00Z", "score": 5, "author_flair_text": None, "edited": False,
        "replies": [
            {
                "id": "default_post1-c1-c1", "type": "comment", "parent_id": "default_post1-c1",
                "author": "UserZ",
                "body": "Good question, UserX! I was wondering the same thing, especially how it applies to Z.",
                "timestamp": "2024-01-02T10:05:00Z", "score": 3, "author_flair_text": "Expert", "edited": False,
                "replies": []
            }
        ]
    }]
    parent_of_default_reply_info = ("a comment by 'UserZ' (ID: default_post1-c1-c1, Score: 3) "
                                    "which says: \"Good question, UserX! I was wondering the same thing, especially how it applies to Z.\"")

    prompt_text = (
        "### FEW-SHOT EXAMPLE ###\n"
        "Consider the following dialogue tree context from a previous post (the author's target reply is missing, and any subsequent replies in that specific branch are truncated):\n"
        f"{json.dumps(default_dialogue_tree_context, indent=2)}\n"
        f"The author of that previous post, 'AuthorOP_Default', made a reply to {parent_of_default_reply_info}.\n"
        "The actual reply body from 'AuthorOP_Default' was:"
    )
    true_reply_body = "Thanks UserZ! That's an excellent point regarding Z, and here's my thought on it: ..."
    return {"prompt_segment": prompt_text, "true_reply_body_for_few_shot": true_reply_body}


# --- Main Processing ---
def main():
    global skipped_post_count
    if not os.path.exists(INPUT_JSON_FILE):
        print(f"Error: Input file '{INPUT_JSON_FILE}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
        all_user_data = json.load(f)

    # Flatten all posts from all users into a single list for easier indexing for context shifting
    all_posts_flat_list_with_author = []
    user_g_idx_flat = 0
    for author_name, posts_by_author in all_user_data.items():
        user_g_idx_flat += 1
        post_user_idx_flat = 0
        for post_data in posts_by_author:
            post_user_idx_flat += 1
            # Add a temporary unique ID for this stage if needed, or rely on list index
            all_posts_flat_list_with_author.append({
                "author_name": author_name,
                "original_post_data": post_data,
                "temp_global_id": f"flat_u{user_g_idx_flat}p{post_user_idx_flat}"  # For reference
            })

    total_num_posts_available = len(all_posts_flat_list_with_author)
    if total_num_posts_available == 0:
        print("No posts found in the input file. Exiting.", file=sys.stderr)
        sys.exit(1)

    output_prompts = []
    default_few_shot_details = get_default_few_shot_for_body_prediction()

    processed_post_count = 0

    for current_post_idx, current_post_entry in enumerate(all_posts_flat_list_with_author):
        processed_post_count += 1
        print(f"Processing post {processed_post_count}/{total_num_posts_available}...", end='\r')

        main_author_name = current_post_entry["author_name"]
        current_post_data_original_ref = current_post_entry["original_post_data"]
        # Use a unique prefix based on the original author and its position in their list,
        # or the global temp_global_id if that's more robust. Let's try to find original index.
        # For simplicity, using current_post_idx for the prefix, but ensure it's distinct.
        # A more robust way would be to re-index based on author's original list.
        # Let's assume current_post_idx is sufficient for a unique enough prefix for now.
        current_post_id_prefix = f"orig_idx{current_post_idx}"

        # --- Process current post for target reply ---
        current_post_for_processing = deepcopy(current_post_data_original_ref)
        flat_nodes_of_current_post = generate_unique_ids_for_post(current_post_for_processing, current_post_id_prefix)
        nodes_map_for_current_post = {node['id']: node for node in flat_nodes_of_current_post}
        author_replies_chronological_current = find_author_replies_for_body_prediction(flat_nodes_of_current_post,
                                                                                       main_author_name)

        if len(author_replies_chronological_current) < 1:  # Need at least one reply to be the target
            skipped_post_count += 1
            continue

        target_reply_obj = author_replies_chronological_current[0]  # Most recent reply by author is the target
        cleaned_target_body = clean_text(target_reply_obj.get('body', ""))
        if cleaned_target_body == REMOVED_BODY_MARKER or cleaned_target_body == DELETED_BODY_MARKER or not cleaned_target_body:
            skipped_post_count += 1
            continue
        true_label_body = cleaned_target_body

        # Basic filter: current post's author must have made at least one reply.
        # More complex filters (like num_internal_trees, removed_percentage) are applied to the *context* post.

        # --- Determine the context post (post_idx + 200, wrapped around) ---
        context_post_idx = (current_post_idx + CONTEXT_SHIFT_OFFSET) % total_num_posts_available
        context_post_entry = all_posts_flat_list_with_author[context_post_idx]

        context_author_name = context_post_entry["author_name"]
        context_post_data_original_ref = context_post_entry["original_post_data"]
        context_post_id_prefix = f"ctx_idx{context_post_idx}"  # Unique prefix for context post

        context_post_for_processing = deepcopy(context_post_data_original_ref)
        flat_nodes_of_context_post = generate_unique_ids_for_post(context_post_for_processing, context_post_id_prefix)
        # nodes_map_for_context_post = {node['id']: node for node in flat_nodes_of_context_post} # Not strictly needed here
        author_replies_chronological_context = find_author_replies_for_body_prediction(flat_nodes_of_context_post,
                                                                                       context_author_name)

        # Filters for the *context* post (for few-shot and main context generation)
        if len(author_replies_chronological_context) < 2:  # Context post needs at least 2 replies for few-shot + main context example base
            skipped_post_count += 1
            # print(f"Skipping current_post_idx {current_post_idx} due to insufficient replies in context_post_idx {context_post_idx} (<2)")
            continue

        # Check for removed/deleted in the chosen few-shot reply from context post
        context_few_shot_candidate_reply = author_replies_chronological_context[
            1]  # 2nd most recent from context post for few-shot
        cleaned_context_few_shot_body = clean_text(context_few_shot_candidate_reply.get('body', ''))
        if cleaned_context_few_shot_body == REMOVED_BODY_MARKER or cleaned_context_few_shot_body == DELETED_BODY_MARKER or not cleaned_context_few_shot_body:
            skipped_post_count += 1
            # print(f"Skipping current_post_idx {current_post_idx} due to removed/empty few-shot candidate in context_post_idx {context_post_idx}")
            continue

        num_internal_trees_context = count_qualifying_dialogue_trees(flat_nodes_of_context_post, context_author_name)
        if num_internal_trees_context <= 1:  # Context post should have some interaction richness
            skipped_post_count += 1
            # print(f"Skipping current_post_idx {current_post_idx} due to insufficient internal trees ({num_internal_trees_context}) in context_post_idx {context_post_idx}")
            continue

        # Check removed percentage in the dialogue tree of the *few-shot reply from the context post*
        root_post_actual_id_context = context_post_for_processing.get('id')
        nodes_map_for_context_post_temp = {node['id']: node for node in flat_nodes_of_context_post}  # For get_tlc_id...

        context_few_shot_tlc_id = get_tlc_id_for_target_reply(context_few_shot_candidate_reply,
                                                              nodes_map_for_context_post_temp,
                                                              root_post_actual_id_context)
        if context_few_shot_tlc_id:
            relevant_tree_nodes_context_fs = get_nodes_in_dialogue_tree(flat_nodes_of_context_post,
                                                                        context_few_shot_tlc_id)
            removed_percentage_fs, total_nodes_in_tree_fs = calculate_removed_percentage_in_tree(
                relevant_tree_nodes_context_fs)
            if total_nodes_in_tree_fs > 0 and removed_percentage_fs > 0.33:
                skipped_post_count += 1
                # print(f"Skipping current_post_idx {current_post_idx} due to high removed percentage ({removed_percentage_fs:.2f}) in few-shot tree of context_post_idx {context_post_idx}")
                continue
        # else:
        # If TLC for few-shot reply is not found, it's an issue. For now, we might use default or skip.
        # print(f"Warning: TLC for few-shot reply not found in context_post_idx {context_post_idx}. Current_post_idx {current_post_idx}")
        # This could happen if the few-shot reply itself is a TLC, or structure is unexpected.
        # Depending on strictness, could skip or fall back to default few-shot.
        # For now, if get_truncated_dialogue_tree_for_body_fewshot handles it by returning None, default will be used.

        # --- Generate Few-Shot Example from the CONTEXT post ---
        current_few_shot_prompt_segment = default_few_shot_details["prompt_segment"]
        current_few_shot_true_answer = default_few_shot_details["true_reply_body_for_few_shot"]

        # Use context_few_shot_candidate_reply (which is [1] from context post's replies)
        fs_prompt_seg, fs_true_ans = get_truncated_dialogue_tree_for_body_fewshot(
            flat_nodes_of_context_post,  # Use flat nodes of the context post
            context_few_shot_candidate_reply,  # The chosen reply from context post
            context_author_name  # Author of the context post
        )
        if fs_prompt_seg and fs_true_ans:
            current_few_shot_prompt_segment = fs_prompt_seg
            current_few_shot_true_answer = fs_true_ans
        # else:
        # print(f"Note: Using default few-shot for current_post_idx {current_post_idx} as context-derived one failed for context_post_idx {context_post_idx}")

        # --- Prepare Main Task Context from the CONTEXT post ---
        # The "main task" for the LLM (in terms of context structure) will be based on the *context* post's
        # *most recent* reply by its author. We need to build a context *as if* we are predicting THAT reply.

        context_main_task_target_reply_obj = author_replies_chronological_context[
            0]  # Most recent reply from context post
        cleaned_context_main_task_target_body = clean_text(context_main_task_target_reply_obj.get('body', ''))

        if cleaned_context_main_task_target_body == REMOVED_BODY_MARKER or \
                cleaned_context_main_task_target_body == DELETED_BODY_MARKER or \
                not cleaned_context_main_task_target_body:
            skipped_post_count += 1
            # print(f"Skipping current_post_idx {current_post_idx} because main target for context post (idx {context_post_idx}) is removed/empty.")
            continue

        # Check removed percentage for the main target reply's tree in the *context* post
        context_main_task_target_tlc_id = get_tlc_id_for_target_reply(context_main_task_target_reply_obj,
                                                                      nodes_map_for_context_post_temp,
                                                                      root_post_actual_id_context)
        if context_main_task_target_tlc_id:
            relevant_tree_nodes_context_main = get_nodes_in_dialogue_tree(flat_nodes_of_context_post,
                                                                          context_main_task_target_tlc_id)
            removed_percentage_main, total_nodes_in_tree_main = calculate_removed_percentage_in_tree(
                relevant_tree_nodes_context_main)
            if total_nodes_in_tree_main > 0 and removed_percentage_main > 0.33:
                skipped_post_count += 1
                # print(f"Skipping current_post_idx {current_post_idx} due to high removed percentage ({removed_percentage_main:.2f}) in main target tree of context_post_idx {context_post_idx}")
                continue

        # Build the pruned context from the *context_post_data_original_ref* (hierarchical)
        # by removing the *context_main_task_target_reply_obj*
        main_task_context_hierarchical_raw = deepcopy(
            context_post_data_original_ref)  # Use original hierarchical structure
        main_task_context_pruned_hierarchical = build_post_context_excluding_node_id(
            main_task_context_hierarchical_raw,
            context_main_task_target_reply_obj['id']  # ID of the reply to remove from context post's data
        )
        if not main_task_context_pruned_hierarchical:  # Should not happen unless removing post itself
            skipped_post_count += 1
            # print(f"Skipping current_post_idx {current_post_idx} due to pruned context being None for context_post_idx {context_post_idx}")
            continue

        main_task_context_pruned_serializable = convert_datetimes_to_strings_in_obj(
            main_task_context_pruned_hierarchical)
        main_task_context_json_str = json.dumps(main_task_context_pruned_serializable, indent=2)

        # --- Information about the reply that the LLM should "predict" (based on context post's structure) ---
        # This is about the *context_main_task_target_reply_obj* (the one removed from context_post for main_task_context_json_str)
        # We are telling the LLM to predict a reply for *this* situation.
        # However, the *true_label* will be from the *current_post_idx*.

        parent_of_context_main_target_node = nodes_map_for_context_post_temp.get(
            context_main_task_target_reply_obj['_parent_id'])
        parent_info_for_llm_task_str = "a comment"
        if parent_of_context_main_target_node:
            parent_author_llm = parent_of_context_main_target_node.get('author', 'Unknown User')
            parent_body_llm = clean_text(parent_of_context_main_target_node.get('body', ''))
            parent_id_llm = parent_of_context_main_target_node.get('id', 'unknown_id')
            parent_score_llm = parent_of_context_main_target_node.get('score', 'N/A')  # Fetch score
            parent_info_for_llm_task_str = (f"a comment by '{parent_author_llm}' "
                                            f"(ID: {parent_id_llm}, Score: {parent_score_llm}) "
                                            f"which says: \"{parent_body_llm}\"")

        # The author for the main task prompt is the author of the *context post*.
        main_task_author_for_prompt = context_author_name

        # --- Construct the final prompt for the LLM ---
        # The LLM is asked to predict a reply as if it's for the context post's situation.
        # The actual target reply (true_label_body) is from the *current* post.

        # Get subreddit from the *current* post for saving.
        current_subreddit_for_output = clean_text(current_post_data_original_ref.get('__sub__'))

        # Information about the reply whose body we are *actually* trying to predict (from the *current* post)
        # This is for our records and for the LLM's "instruction" part, distinct from the context it sees.
        actual_target_parent_node = nodes_map_for_current_post.get(target_reply_obj['_parent_id'])
        actual_target_parent_info_str = "a comment"
        actual_target_parent_author = "Unknown User"
        actual_target_parent_body = ""
        actual_target_parent_id = "unknown_id"
        actual_target_parent_score = "N/A"

        if actual_target_parent_node:
            actual_target_parent_author = actual_target_parent_node.get('author', 'Unknown User')
            actual_target_parent_body = clean_text(actual_target_parent_node.get('body', ''))
            actual_target_parent_id = actual_target_parent_node.get('id', 'unknown_id')
            actual_target_parent_score = actual_target_parent_node.get('score', 'N/A')
            actual_target_parent_info_str = (f"a comment by '{actual_target_parent_author}' "
                                             f"(ID: {actual_target_parent_id}, Score: {actual_target_parent_score}) "
                                             f"which says: \"{actual_target_parent_body}\"")

        # Natural language instruction for the LLM.
        # The prompt tells the LLM about the *context post's* situation.
        # The true label is from the *current post*.

        llm_instruction = (
            f"You are provided with a general conversation context from a social media post (under 'CONTEXT FOR PREDICTION TASK'). "
            f"Now, focusing on a specific task: the author of the original post you are *actually* working with is '{main_author_name}'. "
            f"This author, '{main_author_name}', is about to reply to {actual_target_parent_info_str}. " # 使用 current_post 的父评论信息
            f"Your goal is to predict the body of the reply that '{main_author_name}' would write. "
            f"The reply should directly address the content of the comment: \"{actual_target_parent_body}\". " # 使用 current_post 的父评论内容
            f"Please generate only the predicted reply body."
        )

        final_prompt_text = (
            f"{current_few_shot_prompt_segment}\n{current_few_shot_true_answer}\n\n"
            "### CONTEXT FOR PREDICTION TASK ###\n"
            "Below is the conversation context (intended as a general stylistic and topical guide from a different, unrelated post):\n" # 更明确地说明上下文的用途
            f"{main_task_context_json_str}\n\n"
            "### PREDICTION TASK ###\n"
            f"{llm_instruction}\n"
            f"Predicted reply body from '{main_author_name}':" # 明确 LLM 需要模拟 current_post 的作者
        )

        output_prompts.append({
            "user": main_author_name,
            "prompt": final_prompt_text,
            "true_label": true_label_body,
            "subreddit": current_subreddit_for_output,
            "target_reply_parent_author": actual_target_parent_author,
            "target_reply_parent_body": actual_target_parent_body,
            "target_reply_parent_id": actual_target_parent_id,
            "target_reply_parent_score": actual_target_parent_score,
            "context_post_author_for_prompt": context_author_name, # context_post 的作者名仍保留，仅供记录
            # 以下 context_post_target_parent_* 字段可能不再那么直接相关于 LLM 的任务描述，
            # 但可以保留用于分析上下文与实际任务的差异
            "context_post_target_parent_author": parent_of_context_main_target_node.get('author', 'Unknown User') if parent_of_context_main_target_node else 'N/A',
            "context_post_target_parent_body": clean_text(parent_of_context_main_target_node.get('body', '')) if parent_of_context_main_target_node else 'N/A',
            "context_post_target_parent_id": parent_of_context_main_target_node.get('id', 'unknown_id') if parent_of_context_main_target_node else 'N/A',
            "context_post_target_parent_score": parent_of_context_main_target_node.get('score', 'N/A') if parent_of_context_main_target_node else 'N/A',
            "context_post_used_idx_global": context_post_idx,
            "current_post_idx_global": current_post_idx
        })

    with open(WITH_PSEUDO_RANDOM_CONVO_OUTPUT_JSONL, 'w', encoding='utf-8') as f_out:
        for item in output_prompts:
            f_out.write(json.dumps(item) + '\n')

    print(f"\nProcessing complete.")
    print(f"Generated {len(output_prompts)} prompts for '{WITH_PSEUDO_RANDOM_CONVO_OUTPUT_JSONL}'.")
    print(f"Skipped {skipped_post_count} posts/contexts due to filtering criteria.")


if __name__ == '__main__':
    if not os.path.exists(INPUT_JSON_FILE):
        print(f"CRITICAL ERROR: Input file '{INPUT_JSON_FILE}' not found. Please create it or check the path.")
        sys.exit(1)
    main()