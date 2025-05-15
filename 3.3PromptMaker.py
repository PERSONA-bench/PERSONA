import json
import os
import sys
from datetime import datetime, timezone
from copy import deepcopy
import re
from collections import deque

# --- Configuration ---
INPUT_JSON_FILE = "merged_posts_remapped.json"
WITH_CONVO_OUTPUT_JSONL = "WithConversationPrompts_.jsonl"
WITHOUT_CONVO_OUTPUT_JSONL = "WithoutConversationPrompts_.jsonl"

skipped_post_count = 0


# --- Helper Functions ---

def clean_text(text):
    """Basic text cleaning."""
    if text is None:
        return ""
    return str(text).replace("\n", " ").replace("\r", " ").strip()


def parse_timestamp(ts_str):
    """Parses a timestamp string into a datetime object. Handles potential 'Z' for UTC."""
    if not ts_str:
        return None
    try:
        if isinstance(ts_str, datetime):  # Already a datetime object
            # Ensure it's offset-aware and UTC for consistent comparison
            return ts_str.astimezone(timezone.utc) if ts_str.tzinfo else ts_str.replace(tzinfo=timezone.utc)

        original_ts_str = ts_str  # Keep original for fallback or debugging
        if ts_str.endswith('Z'):
            ts_str = ts_str[:-1] + '+00:00'

        # Attempt ISO 8601 directly
        dt_obj = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))  # Redundant replace if handled, but safe
        return dt_obj.astimezone(timezone.utc) if dt_obj.tzinfo else dt_obj.replace(tzinfo=timezone.utc)

    except ValueError:
        try:
            # Try common format like 'YYYY-MM-DD HH:MM:SS'
            dt_obj = datetime.strptime(original_ts_str, '%Y-%m-%d %H:%M:%S')
            return dt_obj.replace(tzinfo=timezone.utc)  # Assume UTC
        except ValueError:
            # Fallback for other possible datetime-like strings if fromisoformat is too strict
            # This is a bit of a guess, could be refined if more formats appear
            try:
                # Example: "2024-07-22T13:49:55" (no Z or offset)
                dt_obj = datetime.strptime(original_ts_str, '%Y-%m-%dT%H:%M:%S')
                return dt_obj.replace(tzinfo=timezone.utc)  # Assume UTC
            except ValueError:
                # print(f"Warning: Could not parse timestamp: {original_ts_str}", file=sys.stderr) # Reduce noise
                return None


def datetime_to_string(dt_obj):
    """Converts a datetime object to an ISO 8601 string with 'Z' for UTC."""
    if isinstance(dt_obj, datetime):
        # Ensure it's UTC before formatting
        if dt_obj.tzinfo is None or dt_obj.tzinfo.utcoffset(dt_obj) is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        else:
            dt_obj = dt_obj.astimezone(timezone.utc)
        return dt_obj.isoformat().replace('+00:00', 'Z')
    return dt_obj  # Return as is if not a datetime object


def convert_datetimes_to_strings_in_obj(obj):
    """Recursively converts datetime objects in a list/dict structure to ISO strings."""
    if isinstance(obj, list):
        return [convert_datetimes_to_strings_in_obj(item) for item in obj]
    elif isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if isinstance(v, datetime):
                new_dict[k] = datetime_to_string(v)
            # Remove internal keys like '_parsed_timestamp' before final serialization for prompts
            elif k == '_parsed_timestamp' or k.startswith('_parent'):  # Also removing other internal keys
                continue  # Skip internal helper keys
            else:
                new_dict[k] = convert_datetimes_to_strings_in_obj(v)
        return new_dict
    return obj


def generate_unique_ids_for_post(post_data, post_index_prefix):
    flat_nodes = []
    post_id_str = str(post_index_prefix)
    post_data['id'] = post_id_str
    post_data['type'] = 'post'
    post_data.setdefault('title', None)
    post_data.setdefault('url', None)
    post_data.setdefault('content', None)
    post_data.setdefault('__sub__', None)
    post_data.setdefault('score', None)
    post_data.setdefault('author', None)
    # Convert timestamp to string here if it's a datetime object from original data
    if isinstance(post_data.get('timestamp'), datetime):
        post_data['timestamp'] = datetime_to_string(post_data['timestamp'])
    else:
        post_data.setdefault('timestamp', None)

    flat_nodes.append(post_data)
    comment_counter = 1

    def _assign_ids_recursive(comments_list, parent_id_str_rec):
        nonlocal comment_counter
        if not isinstance(comments_list, list):
            return
        for comment in comments_list:
            if not isinstance(comment, dict):
                continue
            current_id_str = f"{parent_id_str_rec}-c{comment_counter}"
            comment_counter += 1
            comment['id'] = current_id_str
            comment['parent_id'] = parent_id_str_rec
            comment['type'] = 'comment'
            comment.setdefault('body', None)
            comment.setdefault('author', None)
            comment.setdefault('score', None)
            # Convert timestamp to string here
            if isinstance(comment.get('timestamp'), datetime):
                comment['timestamp'] = datetime_to_string(comment['timestamp'])
            else:
                comment.setdefault('timestamp', None)
            flat_nodes.append(comment)
            if 'replies' in comment and isinstance(comment['replies'], list):
                _assign_ids_recursive(comment['replies'], current_id_str)

    _assign_ids_recursive(post_data.get('comments', []), post_id_str)
    return flat_nodes


def find_author_replies_in_post_with_details(flat_nodes_list, post_author_name):
    author_replies = []
    nodes_by_id = {node['id']: node for node in flat_nodes_list}
    for node in flat_nodes_list:
        if node.get('type') == 'comment' and node.get('author') == post_author_name:
            parent_id = node.get('parent_id')
            parent_node = nodes_by_id.get(parent_id)
            parent_body = ""
            parent_author = ""
            if parent_node:
                parent_body = parent_node.get('content', parent_node.get('body', ""))
                parent_author = parent_node.get('author', "")

            # Timestamp is already a string in 'node', parse it for internal use
            parsed_ts = parse_timestamp(node.get('timestamp'))
            if parsed_ts and node.get('body') is not None:
                reply_details = deepcopy(node)
                reply_details['_parsed_timestamp'] = parsed_ts  # Keep datetime for sorting
                reply_details['_parent_id'] = parent_id
                reply_details['_parent_body'] = clean_text(parent_body)
                reply_details['_parent_author'] = parent_author
                author_replies.append(reply_details)
    author_replies.sort(key=lambda r: r['_parsed_timestamp'], reverse=True)
    return author_replies


def count_qualifying_dialogue_trees(all_nodes_in_post, post_author_name):
    if not all_nodes_in_post: return 0
    post_node = all_nodes_in_post[0]
    if post_node.get('type') != 'post': return 0
    nodes_map = {node['id']: node for node in all_nodes_in_post}
    top_level_comment_ids = [node['id'] for node in all_nodes_in_post
                             if node.get('type') == 'comment' and node.get('parent_id') == post_node['id']]
    tree_count = 0
    for tlc_id in top_level_comment_ids:
        author_participated_in_thread = False
        q_thread_nodes = deque()
        # Find the tlc_node object to start BFS
        tlc_node_obj = nodes_map.get(tlc_id)
        if not tlc_node_obj: continue
        q_thread_nodes.append(tlc_node_obj)  # Add the actual node object

        visited_in_thread_check = {tlc_id}  # Store IDs of visited nodes

        while q_thread_nodes:
            current_node_in_thread = q_thread_nodes.popleft()
            if not current_node_in_thread: continue
            if current_node_in_thread.get('author') == post_author_name:
                author_participated_in_thread = True
                break

            current_node_id_for_replies = current_node_in_thread.get('id')
            for node_in_map in all_nodes_in_post:  # Check all nodes in the post for replies
                if node_in_map.get('parent_id') == current_node_id_for_replies and \
                        node_in_map['id'] not in visited_in_thread_check:
                    # Get the actual node object from nodes_map to append
                    reply_node_obj = nodes_map.get(node_in_map['id'])
                    if reply_node_obj:
                        q_thread_nodes.append(reply_node_obj)
                    visited_in_thread_check.add(node_in_map['id'])

        if author_participated_in_thread:
            has_replies_to_tlc = any(node.get('parent_id') == tlc_id for node in nodes_map.values())
            if has_replies_to_tlc:
                tree_count += 1
    return tree_count


def build_post_context_excluding_reply(post_data_orig, reply_to_remove_id):
    if not reply_to_remove_id:
        return deepcopy(post_data_orig)
    context_post_data = deepcopy(post_data_orig)

    if 'comments' in context_post_data:
        new_comments = []
        for c in context_post_data.get('comments', []):
            if not isinstance(c, dict): continue  # Skip if not a dict (e.g. None)
            if c.get('id') != reply_to_remove_id:
                new_comments.append(c)
        context_post_data['comments'] = new_comments

    q = deque(context_post_data.get('comments', []))
    while q:
        current_parent_node = q.popleft()
        if not isinstance(current_parent_node, dict): continue

        if 'replies' in current_parent_node and current_parent_node['replies']:
            new_replies_for_parent = []
            for r_child in current_parent_node['replies']:
                if not isinstance(r_child, dict): continue
                if r_child.get('id') != reply_to_remove_id:
                    new_replies_for_parent.append(r_child)
                    q.append(r_child)
            current_parent_node['replies'] = new_replies_for_parent

    return context_post_data


def get_truncated_dialogue_tree_for_fewshot(full_post_flat_nodes, few_shot_reply_obj):
    if not few_shot_reply_obj or not few_shot_reply_obj.get('id'):
        return None

    nodes_map = {node['id']: node for node in full_post_flat_nodes}
    few_shot_reply_id = few_shot_reply_obj['id']

    # 1. Find the Top-Level Comment (TLC) of this few_shot_reply_obj
    path_to_tlc_ids = []
    current_node_id = few_shot_reply_id
    while current_node_id:
        path_to_tlc_ids.insert(0, current_node_id)
        node = nodes_map.get(current_node_id)
        if not node or node.get('type') == 'post':  # Reached post or error
            break
        parent_id = node.get('parent_id')
        parent_node = nodes_map.get(parent_id)
        if not parent_node or parent_node.get('type') == 'post':
            break  # current_node_id is the TLC
        current_node_id = parent_id

    if not path_to_tlc_ids or nodes_map.get(path_to_tlc_ids[0]).get('type') == 'post':
        # This implies few_shot_reply was a post or something went wrong.
        # If few_shot_reply is a TLC, path_to_tlc_ids[0] is its ID.
        first_node_in_path = nodes_map.get(path_to_tlc_ids[0])
        if not first_node_in_path or first_node_in_path.get('type') == 'post':
            # print(f"Warning: Could not trace few_shot_reply {few_shot_reply_id} to its TLC or it is a post.", file=sys.stderr)
            return None
        tlc_id = path_to_tlc_ids[0]
    else:  # Should not happen if logic above is correct
        tlc_id = path_to_tlc_ids[0]

    # 2. Reconstruct the dialogue tree from TLC down to (but not including) few_shot_reply_obj,
    #    or if few_shot_reply_obj is the TLC, just the TLC with no replies.
    #    All replies AFTER the few_shot_reply_obj in its specific branch should be truncated.

    def build_recursive_truncated(node_id, stop_reply_id):
        node_copy = deepcopy(nodes_map.get(node_id))
        if not node_copy: return None

        # Critical: remove internal _parsed_timestamp before serializing
        node_copy.pop('_parsed_timestamp', None)
        node_copy.pop('_parent_id', None)
        node_copy.pop('_parent_body', None)
        node_copy.pop('_parent_author', None)

        if node_id == stop_reply_id:  # If we are at the target few-shot reply, don't include its replies
            node_copy['replies'] = []
            return node_copy  # Return the node itself, but with its replies truncated

        node_copy['replies'] = []  # Default to empty replies

        # Find children of the original node
        children_of_original_node = [n for n_id, n in nodes_map.items() if n.get('parent_id') == node_id]
        # Sort children by their original timestamp to maintain order before truncation
        children_of_original_node.sort(
            key=lambda r: (parse_timestamp(r.get('timestamp')) or datetime.min.replace(tzinfo=timezone.utc)))

        for child_orig_node in children_of_original_node:
            # If this child is the stop_reply_id, we don't process it further (its body is the target)
            # We need to include the path up to the parent of stop_reply_id.
            # The few-shot context shows the parent and its earlier siblings.

            # Check if stop_reply_id is a descendant of child_orig_node or is child_orig_node itself
            is_on_path_or_is_target = False
            q_desc = deque([child_orig_node['id']])
            visited_desc_check = set()
            while q_desc:
                curr_d_id = q_desc.popleft()
                if curr_d_id == stop_reply_id:
                    is_on_path_or_is_target = True
                    break
                if curr_d_id in visited_desc_check: continue
                visited_desc_check.add(curr_d_id)
                for n_map_id, n_map_val in nodes_map.items():
                    if n_map_val.get('parent_id') == curr_d_id:
                        q_desc.append(n_map_id)

            if child_orig_node['id'] == stop_reply_id:
                # We've reached the few-shot target reply. Don't include it in the *context's tree*.
                # The prompt will state "Author replied to [parent of few_shot_target]".
                # So, the context should show up to its parent.
                break  # Stop adding more children to current node_copy

            truncated_child = build_recursive_truncated(child_orig_node['id'], stop_reply_id)
            if truncated_child:
                node_copy['replies'].append(truncated_child)

            if is_on_path_or_is_target:  # If we processed a child that is on the path to the target, stop here for this parent
                break

        return node_copy

    # Build the truncated tree starting from the TLC
    # The context should show up to the parent of the few_shot_reply_obj.
    # The few_shot_reply_obj itself is *not* part of this context tree.

    # Find parent of few_shot_reply_obj
    parent_of_few_shot_id = few_shot_reply_obj.get('_parent_id')  # This is from enhanced reply obj

    # We need to build the tree, and when we encounter the parent_of_few_shot_id,
    # its 'replies' list should only contain items *before* few_shot_reply_id.

    final_truncated_tlc_tree = build_recursive_truncated(tlc_id,
                                                         few_shot_reply_id)  # Pass the reply itself to stop AT it
    if final_truncated_tlc_tree:
        # Now, we need to ensure that in the structure `final_truncated_tlc_tree`,
        # the `few_shot_reply_id` is actually removed from its parent's `replies` list.
        # The `build_recursive_truncated` as written stops processing *children* of `few_shot_reply_id`.
        # It doesn't remove `few_shot_reply_id` itself from its parent's list.

        # Let's try a simpler approach: build the full TLC tree, then prune.
        def get_full_tlc_branch(node_id_start):
            node_copy = deepcopy(nodes_map.get(node_id_start))
            if not node_copy: return None
            node_copy.pop('_parsed_timestamp', None)  # Clean internal keys
            # ... (add other _key removals here) ...

            node_copy['replies'] = []
            children_of_orig = [n for n_id, n in nodes_map.items() if n.get('parent_id') == node_id_start]
            children_of_orig.sort(
                key=lambda r: (parse_timestamp(r.get('timestamp')) or datetime.min.replace(tzinfo=timezone.utc)))
            for child_orig in children_of_orig:
                full_child_branch = get_full_tlc_branch(child_orig['id'])
                if full_child_branch:
                    node_copy['replies'].append(full_child_branch)
            return node_copy

        full_tlc_tree_for_fewshot = get_full_tlc_branch(tlc_id)
        if full_tlc_tree_for_fewshot:
            # Now remove the few_shot_reply_id from this structure
            # This is tricky because the structure is nested.
            # `build_post_context_excluding_reply` works on a full post. We need similar for a tree branch.

            # For the new logic: "only the dialogue tree of the reply... truncate info after reply"
            # The `build_recursive_truncated` aimed to do this. Let's refine its stop condition.
            # It should build the tree up to the parent of `few_shot_reply_id`, and for that parent,
            # only include replies up to (but not including) `few_shot_reply_id`.

            # Simpler: just return the TLC and its first few direct replies, stopping before the one to predict.
            # This might be too simple and lose context.

            # Let's use the full TLC tree and then use `build_post_context_excluding_reply` logic on this smaller tree.
            # The `build_post_context_excluding_reply` takes a post-like structure.
            # We can wrap `full_tlc_tree_for_fewshot` as if it's the only comment in a dummy post.
            dummy_post_for_pruning = {"type": "post", "id": "dummy", "comments": [full_tlc_tree_for_fewshot]}
            pruned_dummy_post = build_post_context_excluding_reply(dummy_post_for_pruning, few_shot_reply_id)
            if pruned_dummy_post and pruned_dummy_post.get("comments"):
                return convert_datetimes_to_strings_in_obj(
                    pruned_dummy_post["comments"])  # Return list of TLCs (just one)

    return None  # Fallback


def get_default_few_shot_truncated():
    default_dialogue_tree_context = [
        {
            "id": "default_tlc1", "type": "comment", "parent_id": "default_post1",
            "author": "UserX", "body": "This is a top-level comment in the default example.",
            "timestamp": "2024-01-02T10:00:00Z", "score": 7,
            "replies": [
                {
                    "id": "default_tlc1-r1", "type": "comment", "parent_id": "default_tlc1",
                    "author": "UserY", "body": "Here's a reply from another user.",
                    # This is the parent of the predicted reply
                    "timestamp": "2024-01-02T10:05:00Z", "score": 4,
                    "replies": []
                }
            ]
        }
    ]
    prompt_text = (
        "### FEW-SHOT EXAMPLE ###\n"
        "Consider the following dialogue tree context (the author's target few-shot reply is missing, and any subsequent replies in that specific branch are truncated):\n"
        f"{json.dumps(default_dialogue_tree_context, indent=2)}\n"
        "The author of the post, 'AuthorOP', replied to the comment with ID 'default_tlc1-r1' (body: \"Here's a reply from another user.\").\n"
        "Predict AuthorOP's reply.\n"
        "Predicted Reply from AuthorOP to 'default_tlc1-r1':"
    )
    true_reply_body = "This is the AuthorOP's insightful reply for the few-shot example."
    return {"prompt_segment": prompt_text, "true_reply_body_for_few_shot": true_reply_body}


# --- Main Processing ---
def main():
    global skipped_post_count
    if not os.path.exists(INPUT_JSON_FILE):
        print(f"Error: Input file '{INPUT_JSON_FILE}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
        all_user_data = json.load(f)

    with_convo_prompts = []
    without_convo_prompts = []

    default_few_shot_details = get_default_few_shot_truncated()

    user_g_idx = 0
    for main_author_name, posts_by_author_orig in all_user_data.items():
        user_g_idx += 1
        author_all_posts_summary_for_without = []
        # Create a deepcopy for iterating and modifying for history to avoid issues with main loop variable
        posts_for_history_generation = deepcopy(posts_by_author_orig)
        for hist_post_data in posts_for_history_generation:
            author_all_posts_summary_for_without.append({
                "title": clean_text(hist_post_data.get('title')),
                "__sub__": clean_text(hist_post_data.get('__sub__')),
                "content": clean_text(hist_post_data.get('content', hist_post_data.get('body', "")))
            })

        post_user_idx = 0
        for current_post_data_original_ref in posts_by_author_orig:  # Iterate original for safety
            post_user_idx += 1
            current_post_id_prefix = f"u{user_g_idx}p{post_user_idx}"

            # Always work with a deepcopy for processing THIS post to assign IDs and prune
            current_post_for_processing = deepcopy(current_post_data_original_ref)
            flat_nodes_of_current_post = generate_unique_ids_for_post(current_post_for_processing,
                                                                      current_post_id_prefix)

            author_replies_in_current_post = find_author_replies_in_post_with_details(flat_nodes_of_current_post,
                                                                                      main_author_name)
            num_author_replies = len(author_replies_in_current_post)
            num_internal_trees = count_qualifying_dialogue_trees(flat_nodes_of_current_post, main_author_name)

            if num_author_replies <= 2 or num_internal_trees <= 1:
                skipped_post_count += 1
                continue

            target_reply_obj = author_replies_in_current_post[0]
            few_shot_target_reply_obj = author_replies_in_current_post[1]

            true_label_body = clean_text(target_reply_obj.get('body'))

            # --- Generate for WithConversationPrompts.jsonl ---
            current_few_shot_prompt_segment = default_few_shot_details["prompt_segment"]
            current_few_shot_true_answer = default_few_shot_details["true_reply_body_for_few_shot"]

            if few_shot_target_reply_obj:
                # `current_post_for_processing` already has IDs.
                # `flat_nodes_of_current_post` is the flat list derived from it.
                truncated_few_shot_tree_list = get_truncated_dialogue_tree_for_fewshot(flat_nodes_of_current_post,
                                                                                       few_shot_target_reply_obj)

                if truncated_few_shot_tree_list:
                    parent_of_few_shot_info = (f"a comment by '{few_shot_target_reply_obj['_parent_author']}' "
                                               f"(ID: {few_shot_target_reply_obj['_parent_id']}) "
                                               f"which says: \"{few_shot_target_reply_obj['_parent_body']}\"")

                    # The truncated_few_shot_tree_list should represent the context *before* the few-shot reply.
                    # The `json.dumps` will use the string-converted timestamps.
                    few_shot_context_json_str = json.dumps(
                        convert_datetimes_to_strings_in_obj(truncated_few_shot_tree_list), indent=2)

                    current_few_shot_prompt_segment = (
                        "### FEW-SHOT EXAMPLE ###\n"
                        "Consider the following dialogue tree context (the author's target few-shot reply is missing, and any subsequent replies in that specific branch are truncated):\n"
                        f"{few_shot_context_json_str}\n"
                        f"The author of the post, '{main_author_name}', replied to {parent_of_few_shot_info}. "
                        f"(The author's reply would have ID: {few_shot_target_reply_obj['id']}).\n"  # Keep ID for reference
                        "Predict the author's reply.\n"
                        f"Predicted Reply from {main_author_name}:"
                    )
                    current_few_shot_true_answer = clean_text(few_shot_target_reply_obj.get('body'))

            main_task_context_raw = deepcopy(current_post_for_processing)
            main_task_context_pruned = build_post_context_excluding_reply(main_task_context_raw, target_reply_obj['id'])
            main_task_context_pruned_serializable = convert_datetimes_to_strings_in_obj(main_task_context_pruned)

            parent_of_main_target_info = (f"a comment by '{target_reply_obj['_parent_author']}' "
                                          f"(ID: {target_reply_obj['_parent_id']}) "
                                          f"which says: \"{target_reply_obj['_parent_body']}\"")

            with_convo_prompt_text = (
                f"{current_few_shot_prompt_segment}\n{current_few_shot_true_answer}\n\n"
                "### ACTUAL TASK ###\n"
                "Now, given the following full post conversation context (the author's target reply is missing):\n"
                f"{json.dumps(main_task_context_pruned_serializable, indent=2)}\n"
                f"The author of the post, '{main_author_name}', replied to {parent_of_main_target_info}. "
                f"(The author's reply would have ID: {target_reply_obj['id']}).\n"
                "Predict the author's reply.\n"
                f"Predicted Reply from {main_author_name}:"
            )

            with_convo_prompts.append({
                "user": main_author_name,  # <--- 添加 user 字段
                "prompt": with_convo_prompt_text,
                "true_label": true_label_body
            })

            # --- Generate for WithoutConversationPrompts.jsonl ---
            history_str_parts = []
            # For identifying the current post to exclude from history
            current_post_orig_title = clean_text(current_post_data_original_ref.get('title', ""))
            current_post_orig_url = current_post_data_original_ref.get('url')  # URL is a better unique ID if available

            for idx, hist_post_summary_data in enumerate(author_all_posts_summary_for_without):
                original_post_for_comparison = posts_by_author_orig[idx]  # Get original post by index

                is_current_post = False
                if current_post_orig_url and original_post_for_comparison.get('url') == current_post_orig_url:
                    is_current_post = True
                elif not current_post_orig_url and \
                        hist_post_summary_data['title'] == current_post_orig_title:  # Fallback to title if no URL
                    # Could add more fields for fallback comparison if needed (e.g., first few chars of content)
                    is_current_post = True

                if not is_current_post:
                    history_str_parts.append(
                        f"- A post titled \"{hist_post_summary_data['title']}\" in the '{hist_post_summary_data['__sub__']}' subreddit. "
                        f"The content was: \"{hist_post_summary_data['content'][:200]}{'...' if len(hist_post_summary_data['content']) > 200 else ''}\"."
                    )

            history_summary = f"The author '{main_author_name}' has no other available post history for context."
            if history_str_parts:
                history_summary = f"Context from author '{main_author_name}'s post history:\n" + "\n".join(
                    history_str_parts)

            # Use current_post_for_processing for current post info as it has IDs and might have been cleaned
            current_post_info_str = (
                f"The current post is titled \"{clean_text(current_post_for_processing.get('title'))}\" "
                f"in the '{clean_text(current_post_for_processing.get('__sub__'))}' subreddit. "
                f"The post content is: \"{clean_text(current_post_for_processing.get('content'))}\"."
            )

            without_convo_prompt_text = (
                f"{history_summary}\n\n"
                f"{current_post_info_str}\n\n"
                f"Within this current post, '{main_author_name}' is about to reply to the following comment made by '{target_reply_obj['_parent_author']}':\n"
                f"\"{target_reply_obj['_parent_body']}\"\n\n"
                f"What will '{main_author_name}' write in their reply?"
            )

            without_convo_prompts.append({
                "user": main_author_name,  # <--- 添加 user 字段
                "prompt": without_convo_prompt_text,
                "true_label": true_label_body
            })

    with open(WITH_CONVO_OUTPUT_JSONL, 'w', encoding='utf-8') as f_with:
        for item in with_convo_prompts:
            f_with.write(json.dumps(item) + '\n')

    with open(WITHOUT_CONVO_OUTPUT_JSONL, 'w', encoding='utf-8') as f_without:
        for item in without_convo_prompts:
            f_without.write(json.dumps(item) + '\n')

    print(f"\nProcessing complete.")
    print(f"Generated {len(with_convo_prompts)} prompts for '{WITH_CONVO_OUTPUT_JSONL}'.")
    print(f"Generated {len(without_convo_prompts)} prompts for '{WITHOUT_CONVO_OUTPUT_JSONL}'.")
    print(f"Skipped {skipped_post_count} posts due to filtering criteria.")


if __name__ == '__main__':
    main()