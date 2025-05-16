import json
import os
import sys
from datetime import datetime, timezone
from copy import deepcopy
import re
from collections import deque

# --- Configuration ---
INPUT_JSON_FILE = "demo/demo.json"
# Output filenames for Task 3.3: Predict Reply Body
WITH_CONVO_OUTPUT_JSONL = "WithConversationPrompts_BodyPrediction_v2.jsonl"
WITHOUT_CONVO_OUTPUT_JSONL = "WithoutConversationPrompts_BodyPrediction_v2.jsonl"

skipped_post_count = 0
REMOVED_BODY_MARKER = "[removed]"
DELETED_BODY_MARKER = "[deleted]"


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
            if for_history and k == 'comments':  # In case 'comments' is still in the dict for current post detail
                continue
            if isinstance(v, datetime):
                new_dict[k] = datetime_to_string(v)
            elif k.startswith('_'):
                if not for_history:
                    if not k.startswith('__'):
                        continue
                elif for_history:  # For history-style objects (and now current post detail obj)
                    # Keep __sub__, __whatever, but remove _parsed_timestamp etc.
                    if k in ['_parsed_timestamp', '_abs_score', '_parent_id', '_parent_body', '_parent_author']:
                        continue  # Explicitly remove these helper fields from the JSON output
                    if not k.startswith('__'):  # if it's like _internal_field, skip
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
        if key == 'content' and post_data.get(key) is None:  # Ensure 'content' key exists, possibly from 'body'
            post_data[key] = post_data.get('body')

    if isinstance(post_data.get('timestamp'), datetime):
        post_data['timestamp'] = datetime_to_string(post_data['timestamp'])

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
            parent_body = parent_node.get('content', parent_node.get('body', "")) if parent_node else ""
            parent_author = parent_node.get('author', "") if parent_node else ""

            parsed_ts = parse_timestamp(node.get('timestamp'))
            if parsed_ts:
                reply_details = deepcopy(node)
                reply_details['_parsed_timestamp'] = parsed_ts
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
        tlc_node_obj = nodes_map.get(tlc_id)
        if not tlc_node_obj: continue
        q_thread_nodes.append(tlc_node_obj)
        visited_in_thread_check = {tlc_id}
        temp_q = deque([tlc_node_obj])
        visited_temp = {tlc_id}
        while temp_q:
            curr_temp = temp_q.popleft()
            if curr_temp.get('author') == post_author_name:
                author_participated_in_thread = True
                break
            for node_in_map in all_nodes_in_post:
                if node_in_map.get('parent_id') == curr_temp.get('id') and node_in_map.get('id') not in visited_temp:
                    reply_obj = nodes_map.get(node_in_map.get('id'))
                    if reply_obj: temp_q.append(reply_obj)
                    visited_temp.add(node_in_map.get('id'))

        if author_participated_in_thread:
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
            return current_id
        if not parent_id:
            return None
        current_node = nodes_map.get(parent_id)
        if not current_node:
            return None
        current_id = current_node.get('id')
    return None


def get_nodes_in_dialogue_tree(all_post_nodes_flat, tlc_id_for_tree):
    tree_nodes_list = []
    if not tlc_id_for_tree:
        return tree_nodes_list
    nodes_to_visit_q = deque()
    tlc_node_obj = next(
        (n for n in all_post_nodes_flat if n.get('id') == tlc_id_for_tree and n.get('type') == 'comment'), None)

    if tlc_node_obj:
        nodes_to_visit_q.append(tlc_node_obj)
        tree_nodes_list.append(tlc_node_obj)
        visited_node_ids = {tlc_id_for_tree}
        while nodes_to_visit_q:
            current_parent_node_tree = nodes_to_visit_q.popleft()
            current_parent_id_tree = current_parent_node_tree.get('id')
            for child_node in all_post_nodes_flat:
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


def build_post_context_excluding_node(post_data_orig, node_to_remove_id):
    if not node_to_remove_id: return deepcopy(post_data_orig)
    context_data = deepcopy(post_data_orig)
    if context_data.get('id') == node_to_remove_id and context_data.get('type') == 'post': return None
    if 'comments' in context_data:
        new_comments = []
        for c_top in context_data.get('comments', []):
            if isinstance(c_top, dict) and c_top.get('id') != node_to_remove_id:
                new_comments.append(c_top)
        context_data['comments'] = new_comments

    q = deque(context_data.get('comments', []))
    while q:
        current_parent_node = q.popleft()
        if not isinstance(current_parent_node, dict): continue
        if 'replies' in current_parent_node and current_parent_node['replies']:
            new_replies = []
            for r_child in current_parent_node['replies']:
                if isinstance(r_child, dict) and r_child.get('id') != node_to_remove_id:
                    new_replies.append(r_child)
            current_parent_node['replies'] = new_replies
            for r_child_ok in current_parent_node['replies']: q.append(r_child_ok)
    return context_data


def get_truncated_dialogue_tree_for_body_fewshot(full_post_flat_nodes, few_shot_reply_obj):
    if not few_shot_reply_obj or not few_shot_reply_obj.get('id'): return None
    nodes_map = {node['id']: node for node in full_post_flat_nodes}
    few_shot_reply_id_to_exclude = few_shot_reply_obj['id']

    tlc_id = None
    current_node_id_trace = few_shot_reply_id_to_exclude
    path_to_tlc = [few_shot_reply_id_to_exclude]
    while current_node_id_trace:
        node_trace = nodes_map.get(current_node_id_trace)
        if not node_trace or node_trace.get('type') == 'post': break
        parent_id_trace = node_trace.get('parent_id')
        parent_node_trace = nodes_map.get(parent_id_trace)
        if not parent_node_trace or parent_node_trace.get('type') == 'post':
            tlc_id = current_node_id_trace
            break
        path_to_tlc.append(parent_id_trace)
        current_node_id_trace = parent_id_trace
    if not tlc_id: return None
    path_to_tlc.reverse()

    def build_display_branch_up_to_parent(node_id_current, id_of_reply_to_exclude_branch, exclusion_path):
        original_node = nodes_map.get(node_id_current)
        if not original_node: return None

        node_copy_display = deepcopy(original_node)
        node_copy_display.pop('replies', None)
        node_copy_display['replies'] = []

        children_original = [n for n in full_post_flat_nodes if n.get('parent_id') == node_id_current]
        children_original.sort(
            key=lambda r: (parse_timestamp(r.get('timestamp')) or datetime.min.replace(tzinfo=timezone.utc)))

        for child_node_orig in children_original:
            if id_of_reply_to_exclude_branch and child_node_orig['id'] == id_of_reply_to_exclude_branch:
                break

            is_on_path_to_excluded = False
            if id_of_reply_to_exclude_branch and child_node_orig['id'] in exclusion_path:
                try:
                    current_idx_in_path = exclusion_path.index(node_id_current)
                    if current_idx_in_path + 1 < len(exclusion_path) and exclusion_path[current_idx_in_path + 1] == \
                            child_node_orig['id']:
                        is_on_path_to_excluded = True
                except ValueError:
                    pass
            branch_for_child = build_display_branch_up_to_parent(
                child_node_orig['id'],
                id_of_reply_to_exclude_branch if is_on_path_to_excluded else None,
                exclusion_path if is_on_path_to_excluded else []
            )
            if branch_for_child:
                node_copy_display['replies'].append(branch_for_child)
        return node_copy_display

    truncated_tlc_tree_for_fewshot_context = build_display_branch_up_to_parent(tlc_id, few_shot_reply_id_to_exclude,
                                                                               path_to_tlc)
    return [convert_datetimes_to_strings_in_obj(
        truncated_tlc_tree_for_fewshot_context)] if truncated_tlc_tree_for_fewshot_context else None


def get_default_few_shot_for_body_prediction():
    default_dialogue_tree_context = [{
        "id": "default_tlc1", "type": "comment", "parent_id": "default_post1",
        "author": "UserX", "body": "This is a great post about topic Y. I have a follow-up question.",
        "timestamp": "2024-01-02T10:00:00Z", "score": 5,
        "replies": [
            {
                "id": "default_tlc1-r1", "type": "comment", "parent_id": "default_tlc1",
                "author": "UserZ", "body": "Good question, UserX! I was wondering the same thing.",
                "timestamp": "2024-01-02T10:05:00Z", "score": 3,
                "replies": []
            }
        ]
    }]
    prompt_text = (
        "### FEW-SHOT EXAMPLE ###\n"
        "Consider the following dialogue tree context from a post (the author's target reply is missing, and any subsequent replies in that specific branch are truncated):\n"
        f"{json.dumps(default_dialogue_tree_context, indent=2)}\n"
        "The author of the post, 'AuthorOP', made a reply to the comment with ID 'default_tlc1-r1' (body: \"Good question, UserX! I was wondering the same thing.\").\n"
        "Predict the body text of AuthorOP's reply to 'default_tlc1-r1'.\n"
        "Predicted body text for AuthorOP's reply:"
    )
    true_reply_body = "Thanks UserZ! That's an excellent point, and here's my thought on it..."
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
    default_few_shot_details = get_default_few_shot_for_body_prediction()
    user_g_idx = 0

    for main_author_name, posts_by_author_orig in all_user_data.items():
        user_g_idx += 1

        # Removed: author_all_posts_history_objects_for_json and complete_history_json_for_this_author
        # as they are no longer used in the without_convo_prompt.

        post_user_idx = 0
        for current_post_data_original_ref in posts_by_author_orig:
            post_user_idx += 1
            current_post_id_prefix = f"u{user_g_idx}p{post_user_idx}"
            current_post_for_processing = deepcopy(current_post_data_original_ref)
            flat_nodes_of_current_post = generate_unique_ids_for_post(current_post_for_processing,
                                                                      current_post_id_prefix)

            nodes_map_for_current_post = {node['id']: node for node in flat_nodes_of_current_post}

            author_replies_chronological = find_author_replies_for_body_prediction(flat_nodes_of_current_post,
                                                                                   main_author_name)
            num_author_replies = len(author_replies_chronological)

            if num_author_replies < 2:
                skipped_post_count += 1
                continue

            target_reply_obj = author_replies_chronological[0]
            cleaned_target_body = clean_text(target_reply_obj.get('body', ""))
            if cleaned_target_body == REMOVED_BODY_MARKER or cleaned_target_body == DELETED_BODY_MARKER:
                skipped_post_count += 1
                continue
            true_label_body = cleaned_target_body

            num_internal_trees = count_qualifying_dialogue_trees(flat_nodes_of_current_post, main_author_name)
            if num_internal_trees <= 1:
                skipped_post_count += 1
                continue

            root_post_actual_id = current_post_for_processing.get('id')
            target_reply_tlc_id = get_tlc_id_for_target_reply(target_reply_obj, nodes_map_for_current_post,
                                                              root_post_actual_id)
            if target_reply_tlc_id:
                relevant_tree_nodes = get_nodes_in_dialogue_tree(flat_nodes_of_current_post, target_reply_tlc_id)
                removed_percentage, total_nodes_in_tree = calculate_removed_percentage_in_tree(relevant_tree_nodes)
                if total_nodes_in_tree > 0 and removed_percentage > 0.33:
                    skipped_post_count += 1
                    continue

            few_shot_target_reply_obj = author_replies_chronological[1]
            current_few_shot_prompt_segment = default_few_shot_details["prompt_segment"]
            current_few_shot_true_answer = default_few_shot_details["true_reply_body_for_few_shot"]

            if few_shot_target_reply_obj:
                truncated_few_shot_tree_context_list = get_truncated_dialogue_tree_for_body_fewshot(
                    flat_nodes_of_current_post,
                    few_shot_target_reply_obj
                )
                if truncated_few_shot_tree_context_list:
                    parent_of_few_shot_info = (f"a comment by '{few_shot_target_reply_obj['_parent_author']}' "
                                               f"(ID: {few_shot_target_reply_obj['_parent_id']}) "
                                               f"which says: \"{few_shot_target_reply_obj['_parent_body']}\"")
                    few_shot_context_json_str = json.dumps(truncated_few_shot_tree_context_list, indent=2)
                    current_few_shot_prompt_segment = (
                        "### FEW-SHOT EXAMPLE ###\n"
                        "Consider the following dialogue tree context (the author's target reply is missing, and subsequent replies in that specific branch are truncated):\n"
                        f"{few_shot_context_json_str}\n"
                        f"The author of the post, '{main_author_name}', made a reply to {parent_of_few_shot_info}. "
                        f"(This reply by '{main_author_name}' had ID: {few_shot_target_reply_obj['id']}).\n"
                        "Predict the body text of this reply by '{main_author_name}'.\n"
                        f"Predicted body text for {main_author_name}'s reply:")
                    current_few_shot_true_answer = clean_text(few_shot_target_reply_obj.get('body'))

            main_task_context_raw = deepcopy(current_post_for_processing)
            main_task_context_pruned = build_post_context_excluding_node(main_task_context_raw, target_reply_obj['id'])
            main_task_context_pruned_serializable = convert_datetimes_to_strings_in_obj(main_task_context_pruned)

            parent_of_main_target_info = (f"a comment by '{target_reply_obj['_parent_author']}' "
                                          f"(ID: {target_reply_obj['_parent_id']}) "
                                          f"which says: \"{target_reply_obj['_parent_body']}\"")
            current_subreddit = clean_text(current_post_for_processing.get('__sub__')) # Get the subreddit
            with_convo_prompt_text = (
                f"{current_few_shot_prompt_segment}\n{current_few_shot_true_answer}\n\n"
                "### ACTUAL TASK ###\n"
                "Now, given the following full post conversation context (the author's target reply is missing):\n"
                f"{json.dumps(main_task_context_pruned_serializable, indent=2)}\n"
                f"The author of the post, '{main_author_name}', made a reply to {parent_of_main_target_info}. "
                f"(This reply by '{main_author_name}' has ID: {target_reply_obj['id']}).\n"
                "Predict the body text of this reply by '{main_author_name}'. Respond ONLY with the predicted text content of the body.\n"
                f"Predicted body text for {main_author_name}'s reply:")
            with_convo_prompts.append(
                {"user": main_author_name,
                 "prompt": with_convo_prompt_text,
                 "true_label": true_label_body,
                 "subreddit": current_subreddit  # Added subreddit key
                })
            # --- Prepare current post details in JSON for without_convo prompt ---
            current_post_detail_for_json_dict = deepcopy(
                current_post_data_original_ref)  # Use original before it's filled with full comment tree for processing
            current_post_detail_for_json_dict.pop('comments', None)  # Remove comments section for this summary

            # Ensure standard fields, similar to how historical posts were processed
            # Standard keys: 'title', 'url', 'content', '__sub__', 'score', 'author', 'timestamp', 'type', 'id'
            for key_to_ensure in ['title', 'url', 'content', '__sub__', 'score', 'author', 'timestamp', 'type']:
                if key_to_ensure == 'content' and key_to_ensure not in current_post_detail_for_json_dict:
                    # Posts might have 'body' or 'selftext' for their main content.
                    # 'generate_unique_ids_for_post' already tries to set 'content' from 'body' for current_post_for_processing
                    # Here we ensure it for current_post_data_original_ref based copy.
                    current_post_detail_for_json_dict[key_to_ensure] = current_post_detail_for_json_dict.get('body',
                                                                                                             current_post_detail_for_json_dict.get(
                                                                                                                 'selftext'))
                current_post_detail_for_json_dict.setdefault(key_to_ensure, None)

            if current_post_detail_for_json_dict.get('type') is None:
                current_post_detail_for_json_dict['type'] = 'post'
            if current_post_detail_for_json_dict.get('author') is None:  # Should be set from input data
                current_post_detail_for_json_dict['author'] = main_author_name

            current_post_detail_for_json_dict[
                'id'] = current_post_id_prefix  # Assign the generated unique ID for this post

            # Convert datetimes and clean up internal fields for the JSON string
            serializable_current_post_detail = convert_datetimes_to_strings_in_obj(current_post_detail_for_json_dict,
                                                                                   for_history=True)
            current_post_detail_json_str = json.dumps(serializable_current_post_detail, indent=2)

            current_post_json_context_section = (
                f"Below is the detailed information for the current post by '{main_author_name}':\n"
                f"{current_post_detail_json_str}\n\n"
            )
            # --- End current post details preparation ---

            parent_comment_body_for_without = target_reply_obj['_parent_body']
            parent_comment_author_for_without = target_reply_obj['_parent_author']

            # Removed: history_section_for_prompt and current_post_info_for_without (prose version)

            without_convo_prompt_text = (
                f"{current_post_json_context_section}"
                f"Within this current post (details provided above, titled \"{clean_text(current_post_for_processing.get('title'))}\"), '{main_author_name}' is about to make a reply to the following comment from '{parent_comment_author_for_without}':\n"
                f"\"{parent_comment_body_for_without}\"\n\n"
                f"Predict the body text of '{main_author_name}'s reply. Respond ONLY with the predicted text content of the body.\n"
                f"Predicted body text for {main_author_name}'s reply:")
            without_convo_prompts.append(
                {"user": main_author_name,
                 "prompt": without_convo_prompt_text,
                 "true_label": true_label_body,
                 "subreddit": current_subreddit  # Added subreddit key
                })
    with open(WITH_CONVO_OUTPUT_JSONL, 'w', encoding='utf-8') as f_with:
        for item in with_convo_prompts: f_with.write(json.dumps(item) + '\n')
    with open(WITHOUT_CONVO_OUTPUT_JSONL, 'w', encoding='utf-8') as f_without:
        for item in without_convo_prompts: f_without.write(json.dumps(item) + '\n')

    print(f"\nProcessing complete.")
    print(f"Generated {len(with_convo_prompts)} prompts for '{WITH_CONVO_OUTPUT_JSONL}'.")
    print(f"Generated {len(without_convo_prompts)} prompts for '{WITHOUT_CONVO_OUTPUT_JSONL}'.")
    print(f"Skipped {skipped_post_count} posts due to filtering criteria.")


if __name__ == '__main__':
    if not os.path.exists(INPUT_JSON_FILE):
        print(f"CRITICAL ERROR: Input file '{INPUT_JSON_FILE}' not found. Please create it or check the path.")
        sys.exit(1)
    main()