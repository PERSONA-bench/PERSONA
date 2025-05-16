import json
import os
import sys
from datetime import datetime, timezone
from copy import deepcopy
from collections import deque
import random  # Added for probabilistic skip

# --- Configuration ---
INPUT_JSON_FILE = "demo\demo.json"
# Output will be a single JSONL file where each line is an object
# containing the post_data_store and the prediction_task.
OUTPUT_JSONL_FILE = "WithConversationPrompts_ScorePrediction_Refactored.jsonl"
# WITHOUT_CONVO_OUTPUT_JSONL will now have a structure similar to OUTPUT_JSONL_FILE
# but with specific content constraints for the "without conversation" context.
WITHOUT_CONVO_OUTPUT_JSONL = "WithoutConversationPrompts_ScorePrediction_Refactored.jsonl"

skipped_post_count = 0
skipped_by_probability_count = 0  # New counter
processed_post_count = 0
deleted_half_positive_low_score_count = 0  # 新增计数器


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
        # Attempt to parse with microseconds
        try:
            dt_obj = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        except ValueError:
            # Attempt to parse without microseconds if the first try fails
            dt_obj = datetime.strptime(ts_str.split('.')[0], '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc)

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
                print(f"Warning: Could not parse timestamp: {original_ts_str}", file=sys.stderr)
                return None


def datetime_to_string(dt_obj):
    if isinstance(dt_obj, datetime):
        if dt_obj.tzinfo is None or dt_obj.tzinfo.utcoffset(dt_obj) is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        else:
            dt_obj = dt_obj.astimezone(timezone.utc)
        return dt_obj.isoformat().replace('+00:00', 'Z')
    return dt_obj


def convert_datetimes_to_strings_in_obj(obj):
    if isinstance(obj, list):
        return [convert_datetimes_to_strings_in_obj(item) for item in obj]
    elif isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if isinstance(v, datetime):
                new_dict[k] = datetime_to_string(v)
            elif k.startswith('_') and k not in ['__sub__']:  # Preserve __sub__
                continue
            else:
                new_dict[k] = convert_datetimes_to_strings_in_obj(v)
        return new_dict
    return obj


def generate_unique_ids_for_post(post_data, post_index_prefix):
    flat_nodes = []
    post_id_str = str(post_index_prefix)
    post_data['id'] = post_id_str
    post_data['type'] = 'post'
    for key in ['title', 'url', 'content', '__sub__', 'score', 'author', 'timestamp', 'comments']:
        post_data.setdefault(key, None if key != 'comments' else [])
    if isinstance(post_data.get('timestamp'), datetime):
        post_data['timestamp'] = datetime_to_string(post_data['timestamp'])

    post_copy_for_flat = deepcopy(post_data)
    if 'comments' in post_copy_for_flat:
        del post_copy_for_flat['comments']
    flat_nodes.append(post_copy_for_flat)

    comment_counter = 1
    q = deque()

    if isinstance(post_data.get('comments'), list):
        for comment in post_data['comments']:
            if isinstance(comment, dict):
                comment['parent_id'] = post_id_str
                q.append(comment)

    processed_comments_for_flat_list = []

    def _assign_ids_recursive(comments_list, parent_id_str_rec):
        nonlocal comment_counter
        if not isinstance(comments_list, list):
            return
        for i, comment in enumerate(comments_list):
            if not isinstance(comment, dict):
                continue

            current_id_str = f"{parent_id_str_rec}-c{comment_counter}"
            comment_counter += 1

            comment['id'] = current_id_str
            comment['parent_id'] = parent_id_str_rec
            comment['type'] = 'comment'
            for key in ['body', 'author', 'score', 'timestamp', 'replies']:
                comment.setdefault(key, None if key != 'replies' else [])
            if isinstance(comment.get('timestamp'), datetime):
                comment['timestamp'] = datetime_to_string(comment['timestamp'])

            comment_copy_for_flat = deepcopy(comment)
            if 'replies' in comment_copy_for_flat:
                del comment_copy_for_flat['replies']
            processed_comments_for_flat_list.append(comment_copy_for_flat)

            if 'replies' in comment and isinstance(comment['replies'], list):
                _assign_ids_recursive(comment['replies'], current_id_str)

    _assign_ids_recursive(post_data.get('comments', []), post_id_str)
    flat_nodes.extend(processed_comments_for_flat_list)
    return flat_nodes


def find_author_replies_in_post_for_score_prediction(flat_nodes_list, post_author_name):
    author_replies = []
    nodes_by_id = {node['id']: node for node in flat_nodes_list}
    post_node = next((node for node in flat_nodes_list if node.get('type') == 'post'), None)
    if not post_node: return []

    for node in flat_nodes_list:
        if node.get('type') == 'comment' and node.get('author') == post_author_name:
            score = node.get('score')
            if not isinstance(score, (int, float)) or \
                    not node.get('body') or \
                    clean_text(node.get('body')).lower() in ['[deleted]', '[removed]', '']:
                continue

            parent_id = node.get('parent_id')
            parent_node = nodes_by_id.get(parent_id)

            if parent_node and parent_node.get('type') == 'post':
                parent_body = parent_node.get('content', "")
                parent_author = parent_node.get('author', "")
                parent_score = parent_node.get('score')
                parent_timestamp = parent_node.get('timestamp')
            elif parent_node and parent_node.get('type') == 'comment':
                parent_body = parent_node.get('body', "")
                parent_author = parent_node.get('author', "")
                parent_score = parent_node.get('score')
                parent_timestamp = parent_node.get('timestamp')
            else:
                continue

            if not parent_body or clean_text(parent_body).lower() in ['[deleted]', '[removed]', '']:
                continue

            parsed_ts = parse_timestamp(node.get('timestamp'))
            if parsed_ts:
                reply_details = deepcopy(node)
                reply_details['_parsed_timestamp'] = parsed_ts
                reply_details['_abs_score'] = abs(score)
                reply_details['_parent_id'] = parent_id
                reply_details['_parent_body'] = clean_text(parent_body)
                reply_details['_parent_author'] = clean_text(parent_author)
                reply_details['_parent_score'] = parent_score
                reply_details['_parent_timestamp'] = parent_timestamp
                author_replies.append(reply_details)

    author_replies.sort(key=lambda r: (r['_abs_score'], r['_parsed_timestamp']), reverse=True)
    return author_replies


def build_dialogue_tree_with_hidden_score(post_data_with_ids, node_id_to_hide_score_of):
    context_data = deepcopy(post_data_with_ids)  # Works on a copy
    q = deque()

    # Check if the post itself is the target (though typically we hide comment scores)
    if context_data.get('id') == node_id_to_hide_score_of and 'score' in context_data:
        context_data['score'] = "[SCORE_TO_PREDICT]"

    # Add top-level comments to the queue
    if 'comments' in context_data and isinstance(context_data.get('comments'), list):
        for comment in context_data['comments']:
            if isinstance(comment, dict): q.append(comment)

    # BFS traversal to find and hide the score of the target node
    while q:
        node = q.popleft()
        if not isinstance(node, dict): continue

        if node.get('id') == node_id_to_hide_score_of:
            if 'score' in node:  # Ensure 'score' key exists
                node['score'] = "[SCORE_TO_PREDICT]"
            # No need to search further in this branch if found, but BFS continues other branches

        if 'replies' in node and isinstance(node['replies'], list):
            for reply in node['replies']:
                if isinstance(reply, dict): q.append(reply)
    return context_data


def get_post_summary_string(post_obj_from_store, main_author_name_for_summary):
    title = clean_text(post_obj_from_store.get('title', "N/A"))
    subreddit = clean_text(post_obj_from_store.get('__sub__', "N/A"))
    author = clean_text(post_obj_from_store.get('author', "N/A"))
    score = post_obj_from_store.get('score', "N/A")  # This is post score, not reply score
    timestamp = datetime_to_string(parse_timestamp(post_obj_from_store.get('timestamp'))) or "N/A"
    content_full = clean_text(post_obj_from_store.get('content', post_obj_from_store.get('body', "N/A")))
    content_summary = content_full[:500] + ('...' if len(content_full) > 500 else '')
    post_id = post_obj_from_store.get('id', "N/A")

    return (
        f"Referenced Post (ID: {post_id}):\n"
        f"- Subreddit: r/{subreddit}\n"
        f"- Author: u/{author}\n"
        f"- Score: {score}\n"  # This refers to the score of the POST itself.
        f"- Timestamp: {timestamp}\n"
        f"- Title: \"{title}\"\n"
        f"- Content Summary: \"{content_summary}\"\n"
        f"(Full post content and dialogue tree are available in the data store under post reference ID '{post_id}')"
    )


def get_default_few_shot_details_refactored(post_data_store_for_default):
    default_post_ref_id = "default_fs_post1"
    author_reply_id_to_hide = f"{default_post_ref_id}-c1-r1-ar1"

    default_post_content = {
        "id": default_post_ref_id, "type": "post", "author": "DefaultAuthorOP",
        "title": "Default Example Post for Few-Shot",
        "content": "This is the content of the default example post. It contains some interesting points about AI ethics and future development. The community engagement was moderate.",
        "score": 75, "timestamp": "2023-01-01T10:00:00Z", "__sub__": "defaultsubreddit",
        "url": f"http://example.com/r/defaultsubreddit/comments/{default_post_ref_id}",
        "comments": [
            {
                "id": f"{default_post_ref_id}-c1", "type": "comment", "parent_id": default_post_ref_id,
                "author": "UserAlpha",
                "body": "Very insightful post, DefaultAuthorOP! I appreciate the detailed analysis.", "score": 12,
                "timestamp": "2023-01-01T11:00:00Z",
                "replies": [
                    {
                        "id": f"{default_post_ref_id}-c1-r1", "type": "comment",
                        "parent_id": f"{default_post_ref_id}-c1",
                        "author": "UserBeta",
                        "body": "I agree. I have a follow-up question regarding the second point on data privacy.",
                        "score": 6,
                        "timestamp": "2023-01-01T12:00:00Z",
                        "replies": [
                            {
                                "id": author_reply_id_to_hide,
                                "type": "comment", "parent_id": f"{default_post_ref_id}-c1-r1",
                                "author": "DefaultAuthorOP",
                                "body": "Thanks for the question, UserBeta! That's a great point. My thoughts are that regulations need to catch up quickly.",
                                "score": "[SCORE_TO_PREDICT]",  # Score is already hidden for the example
                                "timestamp": "2023-01-01T13:00:00Z",
                                "replies": []
                            }
                        ]
                    }
                ]
            }
        ]
    }
    if post_data_store_for_default is None: post_data_store_for_default = {}
    # The default_post_content already has the score hidden, so it's stored correctly.
    post_data_store_for_default[default_post_ref_id] = convert_datetimes_to_strings_in_obj(
        deepcopy(default_post_content))

    fs_post_summary_str = get_post_summary_string(default_post_content, "DefaultAuthorOP")
    fs_parent_author = "UserBeta"
    fs_parent_id_ref = f"{default_post_ref_id}-c1-r1"
    fs_parent_body_summary = "I agree. I have a follow-up question regarding the second point on data privacy."[
                             :100] + "..."
    fs_reply_author_name = "DefaultAuthorOP"
    fs_reply_id_ref = author_reply_id_to_hide
    fs_reply_timestamp = "2023-01-01T13:00:00Z"
    fs_reply_body_summary = "Thanks for the question, UserBeta! That's a great point. My thoughts are that regulations need to catch up quickly."

    fs_true_label = "positive"  # Based on the assumption that DefaultAuthorOP's reply would be positive if it had a score.

    prompt_segment = (
        "### FEW-SHOT EXAMPLE ###\n"
        f"{fs_post_summary_str}\n\n"
        f"The author of the post, '{fs_reply_author_name}', made a reply.\n"
        f"Parent Comment (that '{fs_reply_author_name}' replied to, ID: '{fs_parent_id_ref}'): Author: '{fs_parent_author}', Body Summary: \"{fs_parent_body_summary}\"\n"
        f"Author's Reply (ID: '{fs_reply_id_ref}'): Timestamp: '{fs_reply_timestamp}', Body: \"{fs_reply_body_summary}\"\n"
        f"(The actual score of this Author's Reply '{fs_reply_id_ref}' is hidden. Its full context is in post ID '{default_post_ref_id}' in the data store.)\n\n"
        f"Based on all this information, predict if the score for '{fs_reply_author_name}'s reply (ID: '{fs_reply_id_ref}') was positive or negative.\n"
        "Predicted score sentiment:"
    )
    return {
        "prompt_segment": prompt_segment,
        "true_score_sentiment_for_few_shot": fs_true_label,
        "fs_post_ref_id": default_post_ref_id,
        "fs_reply_id_hidden": fs_reply_id_ref
    }


# --- Main Processing ---
def main():
    global skipped_post_count, processed_post_count, skipped_by_probability_count, deleted_half_positive_low_score_count

    if not os.path.exists(INPUT_JSON_FILE):
        print(f"Error: Input file '{INPUT_JSON_FILE}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
        all_user_data = json.load(f)

    output_data_lines_with_convo = []
    output_data_lines_without_convo = []

    temp_post_data_store_for_default_fs = {}
    default_few_shot = get_default_few_shot_details_refactored(temp_post_data_store_for_default_fs)

    all_potential_positive_samples_to_filter = []

    user_g_idx = 0
    for main_author_name, posts_by_author_orig in all_user_data.items():
        user_g_idx += 1
        post_user_idx = 0
        for current_post_data_original_ref in posts_by_author_orig:
            post_user_idx += 1
            post_reference_id = f"u{user_g_idx}p{post_user_idx}"

            current_post_structured_with_ids = deepcopy(current_post_data_original_ref)
            flat_nodes_of_current_post = generate_unique_ids_for_post(current_post_structured_with_ids,
                                                                      post_reference_id)
            current_post_structured_with_ids = convert_datetimes_to_strings_in_obj(current_post_structured_with_ids)

            author_replies_for_score_pred = find_author_replies_in_post_for_score_prediction(
                flat_nodes_of_current_post, main_author_name
            )

            initial_skip = False
            participants = set(node.get('author') for node in flat_nodes_of_current_post if node.get('author'))
            if len(participants) <= 3:
                initial_skip = True

            if not initial_skip:
                num_total_author_replies = sum(1 for node in flat_nodes_of_current_post if
                                               node.get('type') == 'comment' and
                                               node.get('author') == main_author_name and
                                               node.get('body') and
                                               clean_text(node.get('body')).lower() not in ['', '[deleted]',
                                                                                            '[removed]'])
                if num_total_author_replies <= 2:
                    initial_skip = True

            if not initial_skip and not author_replies_for_score_pred:
                initial_skip = True

            if not initial_skip:
                highest_score_reply = author_replies_for_score_pred[0]
                if highest_score_reply['_abs_score'] <= 3:
                    initial_skip = True
                elif not highest_score_reply['_parent_body'] or \
                        clean_text(highest_score_reply['_parent_body']).lower() in ['[deleted]', '[removed]', '']:
                    initial_skip = True

            if initial_skip:
                skipped_post_count += 1
                continue

            target_reply_obj = author_replies_for_score_pred[0]
            true_label_score_sentiment = "positive" if target_reply_obj['score'] > 0 else "negative"
            main_target_reply_id_ref = target_reply_obj['id']  # Define it here for clarity

            if true_label_score_sentiment == "positive":
                all_potential_positive_samples_to_filter.append({
                    "target_reply_obj": target_reply_obj,
                    "current_post_structured_with_ids": current_post_structured_with_ids,  # Store original for now
                    "post_reference_id": post_reference_id,
                    "main_author_name": main_author_name,
                    "author_replies_for_score_pred": author_replies_for_score_pred,
                    "main_target_reply_id_ref": main_target_reply_id_ref,
                    "original_data_for_output": {
                        "true_label_score_sentiment": true_label_score_sentiment,
                        "current_subreddit": clean_text(current_post_structured_with_ids.get('__sub__', "N/A")),
                    }
                })
            else:  # negative 样本直接处理
                post_data_store_with_convo = {}

                # --- FIX: Process current_post_structured_with_ids to hide the target reply's score ---
                # Use deepcopy to ensure the original current_post_structured_with_ids is not modified
                # if it's needed elsewhere in its original state. build_dialogue_tree_with_hidden_score also deepcopies.
                post_to_store_with_hidden_score = build_dialogue_tree_with_hidden_score(
                    deepcopy(current_post_structured_with_ids),
                    main_target_reply_id_ref
                )
                post_data_store_with_convo[post_reference_id] = post_to_store_with_hidden_score
                # --- END FIX ---

                if default_few_shot["fs_post_ref_id"] not in post_data_store_with_convo:
                    post_data_store_with_convo[default_few_shot["fs_post_ref_id"]] = \
                    temp_post_data_store_for_default_fs[default_few_shot["fs_post_ref_id"]]

                second_highest_score_reply_for_fs = None
                if len(author_replies_for_score_pred) > 1:
                    temp_second_reply = author_replies_for_score_pred[1]
                    if temp_second_reply['_abs_score'] > 0 and \
                            temp_second_reply.get('body') and \
                            clean_text(temp_second_reply.get('body')).lower() not in ['', '[deleted]', '[removed]'] and \
                            temp_second_reply.get('_parent_body') and \
                            clean_text(temp_second_reply.get('_parent_body')).lower() not in ['', '[deleted]',
                                                                                              '[removed]']:
                        second_highest_score_reply_for_fs = temp_second_reply

                fs_prompt_segment = default_few_shot["prompt_segment"]
                fs_true_answer = default_few_shot["true_score_sentiment_for_few_shot"]

                fs_target_reply_local_id_for_output = default_few_shot["fs_reply_id_hidden"]
                fs_post_ref_id_for_output = default_few_shot["fs_post_ref_id"]

                if second_highest_score_reply_for_fs:
                    fs_target_reply_local = second_highest_score_reply_for_fs
                    fs_post_summary_str = get_post_summary_string(current_post_structured_with_ids,
                                                                  main_author_name)  # Uses original for summary
                    fs_parent_author = fs_target_reply_local['_parent_author']
                    fs_parent_id_ref = fs_target_reply_local['_parent_id']
                    fs_parent_body_summary = clean_text(fs_target_reply_local['_parent_body'])[:100] + "..."
                    fs_reply_author_name = main_author_name
                    fs_reply_id_ref = fs_target_reply_local['id']
                    fs_reply_timestamp = datetime_to_string(fs_target_reply_local['_parsed_timestamp'])
                    fs_reply_body_summary = clean_text(fs_target_reply_local['body'])
                    fs_true_label_for_fs_example = "positive" if fs_target_reply_local['score'] > 0 else "negative"

                    fs_prompt_segment = (
                        "### FEW-SHOT EXAMPLE ###\n"
                        f"{fs_post_summary_str}\n\n"
                        f"The author of the post, '{fs_reply_author_name}', made a reply.\n"
                        f"Parent Comment (that '{fs_reply_author_name}' replied to, ID: '{fs_parent_id_ref}'): Author: '{fs_parent_author}', Body Summary: \"{fs_parent_body_summary}\"\n"
                        f"Author's Reply (ID: '{fs_reply_id_ref}'): Timestamp: '{fs_reply_timestamp}', Body: \"{fs_reply_body_summary}\"\n"
                        f"(The actual score of this Author's Reply '{fs_reply_id_ref}' is hidden. Its full context is in post ID '{post_reference_id}' in the data store.)\n\n"
                        f"Based on all this information, predict if the score for '{fs_reply_author_name}'s reply (ID: '{fs_reply_id_ref}') was positive or negative.\n"
                        "Predicted score sentiment:"
                    )
                    fs_true_answer = fs_true_label_for_fs_example
                    fs_target_reply_local_id_for_output = fs_reply_id_ref
                    fs_post_ref_id_for_output = post_reference_id

                main_post_summary_str = get_post_summary_string(current_post_structured_with_ids,
                                                                main_author_name)  # Uses original for summary
                main_parent_author = target_reply_obj['_parent_author']
                main_parent_id_ref_for_prompt = target_reply_obj['_parent_id']  # Renamed to avoid confusion
                main_parent_body_summary = clean_text(target_reply_obj['_parent_body'])[:100] + "..."
                main_target_reply_author_name = main_author_name
                # main_target_reply_id_ref is already defined
                main_target_reply_timestamp = datetime_to_string(target_reply_obj['_parsed_timestamp'])
                main_target_reply_body_summary = clean_text(target_reply_obj['body'])

                prompt_text_for_with_convo_task = (
                    f"{fs_prompt_segment}\n{fs_true_answer}\n\n"
                    "### ACTUAL TASK ###\n"
                    f"{main_post_summary_str}\n\n"
                    f"The author of the post, '{main_target_reply_author_name}', made another reply (or the primary reply we are interested in).\n"
                    f"Parent Comment (that '{main_target_reply_author_name}' replied to, ID: '{main_parent_id_ref_for_prompt}'): Author: '{main_parent_author}', Body Summary: \"{main_parent_body_summary}\"\n"
                    f"Author's Reply (ID: '{main_target_reply_id_ref}'): Timestamp: '{main_target_reply_timestamp}', Body: \"{main_target_reply_body_summary}\"\n"
                    f"(Your task is to predict the score sentiment of this Author's Reply '{main_target_reply_id_ref}'. Its full context is in post ID '{post_reference_id}' in the data store.)\n\n"
                    f"Predict whether the score for '{main_target_reply_author_name}'s reply (ID: '{main_target_reply_id_ref}') will be positive or negative. Respond with only 'positive' or 'negative'.\n"
                    "Predicted score sentiment:"
                )
                output_entry_with_convo = {
                    "post_data_store": post_data_store_with_convo,
                    "prediction_task": {
                        "prompt_text": prompt_text_for_with_convo_task,
                        "target_post_reference_id": post_reference_id,
                        "target_reply_id": main_target_reply_id_ref,
                        "true_label": true_label_score_sentiment,
                        "subreddit": clean_text(current_post_structured_with_ids.get('__sub__', "N/A")),
                        "few_shot_info": {
                            "type": "dynamic" if second_highest_score_reply_for_fs else "default",
                            "fs_post_ref_id": fs_post_ref_id_for_output,
                            "fs_reply_id_hidden_in_example": fs_target_reply_local_id_for_output
                        }
                    }
                }
                output_data_lines_with_convo.append(output_entry_with_convo)

                # --- "Without Conversation" Prompt (for negative samples) - NEW FORMAT ---
                post_data_store_without_convo = {}
                simplified_post_for_store = {
                    key: value for key, value in current_post_structured_with_ids.items()  # Uses original
                    if key not in ['comments', 'replies']
                }
                post_data_store_without_convo[post_reference_id] = simplified_post_for_store

                current_subreddit_val = clean_text(current_post_structured_with_ids.get('__sub__', "N/A"))
                post_title_val = clean_text(current_post_structured_with_ids.get('title', "N/A"))
                post_content_val_full = clean_text(current_post_structured_with_ids.get('content', "N/A"))
                post_content_summary_val = post_content_val_full[:500] + (
                    '...' if len(post_content_val_full) > 500 else '')

                without_convo_prompt_text = (
                    f"Consider the following post:\n"
                    f"- Subreddit: r/{current_subreddit_val}\n"
                    f"- Post ID: {post_reference_id}\n"
                    f"- Author: u/{main_author_name}\n"
                    f"- Title: \"{post_title_val}\"\n"
                    f"- Original Post Content Summary: \"{post_content_summary_val}\"\n\n"
                    f"The author of this post, u/{main_author_name}, posted a reply to a comment under their post. "
                    f"The content of the author's reply is:\n"
                    f"\"{clean_text(target_reply_obj['body'])}\"\n\n"
                    f"Predict whether the score of this author's reply will be positive or negative. "
                    f"Respond with only 'positive' or 'negative'.\n"
                    "Predicted score sentiment:"
                )

                output_entry_without_convo = {
                    "post_data_store": post_data_store_without_convo,
                    "prediction_task": {
                        "prompt_text": without_convo_prompt_text,
                        "target_post_reference_id": post_reference_id,
                        "target_reply_id": main_target_reply_id_ref,
                        "true_label": true_label_score_sentiment,
                        "subreddit": current_subreddit_val,
                    }
                }
                output_data_lines_without_convo.append(output_entry_without_convo)
                processed_post_count += 1

    all_potential_positive_samples_to_filter.sort(key=lambda x: x["target_reply_obj"]["_abs_score"])
    num_to_delete = len(all_potential_positive_samples_to_filter) // 2
    samples_to_keep = all_potential_positive_samples_to_filter[num_to_delete:]
    deleted_half_positive_low_score_count = num_to_delete
    skipped_post_count += num_to_delete

    for sample_data in samples_to_keep:  # These are all positive samples
        target_reply_obj = sample_data["target_reply_obj"]
        # current_post_structured_with_ids here is the original, unmodified version
        current_post_structured_with_ids_orig_for_positive = sample_data["current_post_structured_with_ids"]
        post_reference_id = sample_data["post_reference_id"]
        main_author_name = sample_data["main_author_name"]
        author_replies_for_score_pred_for_sample = sample_data["author_replies_for_score_pred"]
        main_target_reply_id_ref = sample_data["main_target_reply_id_ref"]  # Get it from stored data
        original_data = sample_data["original_data_for_output"]
        true_label_score_sentiment = original_data["true_label_score_sentiment"]
        current_subreddit = original_data["current_subreddit"]

        post_data_store_with_convo = {}
        # --- FIX: Process current_post_structured_with_ids to hide the target reply's score ---
        post_to_store_with_hidden_score = build_dialogue_tree_with_hidden_score(
            deepcopy(current_post_structured_with_ids_orig_for_positive),  # Use the original from sample_data
            main_target_reply_id_ref
        )
        post_data_store_with_convo[post_reference_id] = post_to_store_with_hidden_score
        # --- END FIX ---

        if default_few_shot["fs_post_ref_id"] not in post_data_store_with_convo:
            post_data_store_with_convo[default_few_shot["fs_post_ref_id"]] = temp_post_data_store_for_default_fs[
                default_few_shot["fs_post_ref_id"]]

        second_highest_score_reply_for_fs = None
        if len(author_replies_for_score_pred_for_sample) > 1:
            temp_second_reply = author_replies_for_score_pred_for_sample[1]
            if temp_second_reply['_abs_score'] > 0 and \
                    temp_second_reply.get('body') and \
                    clean_text(temp_second_reply.get('body')).lower() not in ['', '[deleted]', '[removed]'] and \
                    temp_second_reply.get('_parent_body') and \
                    clean_text(temp_second_reply.get('_parent_body')).lower() not in ['', '[deleted]', '[removed]']:
                second_highest_score_reply_for_fs = temp_second_reply

        fs_prompt_segment = default_few_shot["prompt_segment"]
        fs_true_answer = default_few_shot["true_score_sentiment_for_few_shot"]

        fs_target_reply_local_id_for_output = default_few_shot["fs_reply_id_hidden"]
        fs_post_ref_id_for_output = default_few_shot["fs_post_ref_id"]

        if second_highest_score_reply_for_fs:
            fs_target_reply_local = second_highest_score_reply_for_fs
            # Use original post data for generating summary string for few-shot
            fs_post_summary_str = get_post_summary_string(current_post_structured_with_ids_orig_for_positive,
                                                          main_author_name)
            fs_parent_author = fs_target_reply_local['_parent_author']
            fs_parent_id_ref = fs_target_reply_local['_parent_id']
            fs_parent_body_summary = clean_text(fs_target_reply_local['_parent_body'])[:100] + "..."
            fs_reply_author_name = main_author_name
            fs_reply_id_ref = fs_target_reply_local['id']
            fs_reply_timestamp = datetime_to_string(fs_target_reply_local['_parsed_timestamp'])
            fs_reply_body_summary = clean_text(fs_target_reply_local['body'])
            fs_true_label_for_fs_example = "positive" if fs_target_reply_local['score'] > 0 else "negative"

            fs_prompt_segment = (
                "### FEW-SHOT EXAMPLE ###\n"
                f"{fs_post_summary_str}\n\n"
                f"The author of the post, '{fs_reply_author_name}', made a reply.\n"
                f"Parent Comment (that '{fs_reply_author_name}' replied to, ID: '{fs_parent_id_ref}'): Author: '{fs_parent_author}', Body Summary: \"{fs_parent_body_summary}\"\n"
                f"Author's Reply (ID: '{fs_reply_id_ref}'): Timestamp: '{fs_reply_timestamp}', Body: \"{fs_reply_body_summary}\"\n"
                f"(The actual score of this Author's Reply '{fs_reply_id_ref}' is hidden. Its full context is in post ID '{post_reference_id}' in the data store.)\n\n"
                f"Based on all this information, predict if the score for '{fs_reply_author_name}'s reply (ID: '{fs_reply_id_ref}') was positive or negative.\n"
                "Predicted score sentiment:"
            )
            fs_true_answer = fs_true_label_for_fs_example
            fs_target_reply_local_id_for_output = fs_reply_id_ref
            fs_post_ref_id_for_output = post_reference_id

        # Use original post data for generating summary string for actual task
        main_post_summary_str = get_post_summary_string(current_post_structured_with_ids_orig_for_positive,
                                                        main_author_name)
        main_parent_author = target_reply_obj['_parent_author']
        main_parent_id_ref_for_prompt = target_reply_obj['_parent_id']  # Renamed
        main_parent_body_summary = clean_text(target_reply_obj['_parent_body'])[:100] + "..."
        main_target_reply_author_name = main_author_name
        # main_target_reply_id_ref is already defined from sample_data
        main_target_reply_timestamp = datetime_to_string(target_reply_obj['_parsed_timestamp'])
        main_target_reply_body_summary = clean_text(target_reply_obj['body'])

        prompt_text_for_with_convo_task = (
            f"{fs_prompt_segment}\n{fs_true_answer}\n\n"
            "### ACTUAL TASK ###\n"
            f"{main_post_summary_str}\n\n"
            f"The author of the post, '{main_target_reply_author_name}', made another reply (or the primary reply we are interested in).\n"
            f"Parent Comment (that '{main_target_reply_author_name}' replied to, ID: '{main_parent_id_ref_for_prompt}'): Author: '{main_parent_author}', Body Summary: \"{main_parent_body_summary}\"\n"
            f"Author's Reply (ID: '{main_target_reply_id_ref}'): Timestamp: '{main_target_reply_timestamp}', Body: \"{main_target_reply_body_summary}\"\n"
            f"(Your task is to predict the score sentiment of this Author's Reply '{main_target_reply_id_ref}'. Its full context is in post ID '{post_reference_id}' in the data store.)\n\n"
            f"Predict whether the score for '{main_target_reply_author_name}'s reply (ID: '{main_target_reply_id_ref}') will be positive or negative. Respond with only 'positive' or 'negative'.\n"
            "Predicted score sentiment:"
        )
        output_entry_with_convo = {
            "post_data_store": post_data_store_with_convo,
            "prediction_task": {
                "prompt_text": prompt_text_for_with_convo_task,
                "target_post_reference_id": post_reference_id,
                "target_reply_id": main_target_reply_id_ref,
                "true_label": true_label_score_sentiment,
                "subreddit": current_subreddit,
                "few_shot_info": {
                    "type": "dynamic" if second_highest_score_reply_for_fs else "default",
                    "fs_post_ref_id": fs_post_ref_id_for_output,
                    "fs_reply_id_hidden_in_example": fs_target_reply_local_id_for_output
                }
            }
        }
        output_data_lines_with_convo.append(output_entry_with_convo)

        # --- "Without Conversation" Prompt (for positive samples) - NEW FORMAT ---
        post_data_store_without_convo = {}
        simplified_post_for_store = {
            key: value for key, value in current_post_structured_with_ids_orig_for_positive.items()  # Use original
            if key not in ['comments', 'replies']
        }
        post_data_store_without_convo[post_reference_id] = simplified_post_for_store

        post_title_val = clean_text(current_post_structured_with_ids_orig_for_positive.get('title', "N/A"))
        post_content_val_full = clean_text(current_post_structured_with_ids_orig_for_positive.get('content', "N/A"))
        post_content_summary_val = post_content_val_full[:500] + ('...' if len(post_content_val_full) > 500 else '')

        without_convo_prompt_text = (
            f"Consider the following post:\n"
            f"- Subreddit: r/{current_subreddit}\n"
            f"- Post ID: {post_reference_id}\n"
            f"- Author: u/{main_author_name}\n"
            f"- Title: \"{post_title_val}\"\n"
            f"- Original Post Content Summary: \"{post_content_summary_val}\"\n\n"
            f"The author of this post, u/{main_author_name}, posted a reply to a comment under their post. "
            f"The content of the author's reply is:\n"
            f"\"{clean_text(target_reply_obj['body'])}\"\n\n"
            f"Predict whether the score of this author's reply will be positive or negative. "
            f"Respond with only 'positive' or 'negative'.\n"
            "Predicted score sentiment:"
        )

        output_entry_without_convo = {
            "post_data_store": post_data_store_without_convo,
            "prediction_task": {
                "prompt_text": without_convo_prompt_text,
                "target_post_reference_id": post_reference_id,
                "target_reply_id": main_target_reply_id_ref,
                "true_label": true_label_score_sentiment,
                "subreddit": current_subreddit,
            }
        }
        output_data_lines_without_convo.append(output_entry_without_convo)
        processed_post_count += 1

    with open(OUTPUT_JSONL_FILE, 'w', encoding='utf-8') as f_out:
        for item in output_data_lines_with_convo:
            f_out.write(json.dumps(item) + '\n')

    with open(WITHOUT_CONVO_OUTPUT_JSONL, 'w', encoding='utf-8') as f_without:
        for item in output_data_lines_without_convo:
            f_without.write(json.dumps(item) + '\n')

    print(f"\nProcessing complete.")
    print(f"Generated {len(output_data_lines_with_convo)} entries for '{OUTPUT_JSONL_FILE}'.")
    print(f"Generated {len(output_data_lines_without_convo)} entries for '{WITHOUT_CONVO_OUTPUT_JSONL}'.")
    print(f"Processed {processed_post_count} posts successfully (counted after all filtering and processing).")
    print(
        f"Skipped {skipped_post_count} posts due to filtering criteria (includes {skipped_by_probability_count} skipped by 98% probability and {deleted_half_positive_low_score_count} positive low-score samples).")


if __name__ == '__main__':
    if not os.path.exists(INPUT_JSON_FILE):
        print(f"CRITICAL ERROR: Input file '{INPUT_JSON_FILE}' not found. Please create it or check the path.")
        sys.exit(1)
    main()