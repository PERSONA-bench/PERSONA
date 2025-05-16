import json
import os
import sys
from datetime import datetime, timezone
from copy import deepcopy
from collections import deque
import random  # Added for probabilistic skip

# --- Configuration ---
INPUT_JSON_FILE = "merged_posts_remapped.json"
# Output will be a single JSONL file where each line is an object
# containing the post_data_store and the prediction_task.
OUTPUT_JSONL_FILE = "WithConversationPrompts_ExactScorePrediction.jsonl"
# WITHOUT_CONVO_OUTPUT_JSONL will now have a structure similar to OUTPUT_JSONL_FILE
# but with specific content constraints for the "without conversation" context.
WITHOUT_CONVO_OUTPUT_JSONL = "WithoutConversationPrompts_ExactScorePrediction.jsonl"

# User-defined flag for prompting with score sign
# Set to True to hint at the target score's sign in the ACTUAL TASK prompt, False otherwise
PROMPT_SIGN_HINT = True

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
                reply_details['_abs_score'] = abs(score)  # Keep for filtering
                reply_details['_parent_id'] = parent_id
                reply_details['_parent_body'] = clean_text(parent_body)
                reply_details['_parent_author'] = clean_text(parent_author)
                reply_details['_parent_score'] = parent_score
                reply_details['_parent_timestamp'] = parent_timestamp
                author_replies.append(reply_details)

    author_replies.sort(key=lambda r: (r['_abs_score'], r['_parsed_timestamp']), reverse=True)
    return author_replies


def build_dialogue_tree_with_hidden_score(post_data_with_ids, node_id_to_hide_score_of):
    context_data = deepcopy(post_data_with_ids)
    q = deque()
    if context_data.get('id') == node_id_to_hide_score_of and context_data.get('type') == 'post':
        pass  # Post scores are not hidden for this task, only reply scores

    if 'comments' in context_data and isinstance(context_data.get('comments'), list):
        for comment in context_data['comments']:
            if isinstance(comment, dict): q.append(comment)

    while q:
        node = q.popleft()
        if not isinstance(node, dict): continue

        if node.get('id') == node_id_to_hide_score_of:
            if 'score' in node:
                node['score'] = "[SCORE_TO_PREDICT]"  # Keep this placeholder for context

        if 'replies' in node and isinstance(node['replies'], list):
            for reply in node['replies']:
                if isinstance(reply, dict): q.append(reply)
    return context_data


def get_post_summary_string(post_obj_from_store, main_author_name_for_summary):
    title = clean_text(post_obj_from_store.get('title', "N/A"))
    subreddit = clean_text(post_obj_from_store.get('__sub__', "N/A"))
    author = clean_text(post_obj_from_store.get('author', "N/A"))
    score = post_obj_from_store.get('score', "N/A")  # Post score is shown
    timestamp = datetime_to_string(parse_timestamp(post_obj_from_store.get('timestamp'))) or "N/A"
    content_full = clean_text(post_obj_from_store.get('content', post_obj_from_store.get('body', "N/A")))
    content_summary = content_full[:500] + ('...' if len(content_full) > 500 else '')
    post_id = post_obj_from_store.get('id', "N/A")

    return (
        f"Referenced Post (ID: {post_id}):\n"
        f"- Subreddit: r/{subreddit}\n"
        f"- Author: u/{author}\n"
        f"- Score: {score}\n"
        f"- Timestamp: {timestamp}\n"
        f"- Title: \"{title}\"\n"
        f"- Content Summary: \"{content_summary}\"\n"
        f"(Full post content and dialogue tree are available in the data store under post reference ID '{post_id}')"
    )


def get_default_few_shot_details_refactored(post_data_store_for_default):  # For "WithConversation"
    default_post_ref_id = "default_fs_post1_with_convo"
    author_reply_id_to_predict_fs = f"{default_post_ref_id}-c1-r1-ar1"
    default_fs_reply_true_score = 25  # Example numerical score, abs >= 3

    default_post_content = {
        "id": default_post_ref_id, "type": "post", "author": "DefaultAuthorOP",
        "title": "Default Example Post for Few-Shot (With Conversation)",
        "content": "This is the content of the default example post. It contains some interesting points about AI ethics and future development. The community engagement was moderate.",
        "score": 75, "timestamp": "2023-01-01T10:00:00Z", "__sub__": "defaultsubreddit_wc",
        "url": f"http://example.com/r/defaultsubreddit_wc/comments/{default_post_ref_id}",
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
                                "id": author_reply_id_to_predict_fs,
                                # This ID's score is the one we are "predicting" in the example
                                "type": "comment", "parent_id": f"{default_post_ref_id}-c1-r1",
                                "author": "DefaultAuthorOP",
                                "body": "Thanks for the question, UserBeta! That's a great point. My thoughts are that regulations need to catch up quickly.",
                                "score": default_fs_reply_true_score,  # Store the true score here for reference
                                "timestamp": "2023-01-01T13:00:00Z",
                                "replies": []
                            }
                        ]
                    }
                ]
            }
        ]
    }
    default_post_content_for_store = deepcopy(default_post_content)
    q_fs = deque(default_post_content_for_store["comments"])
    while q_fs:
        comment_fs = q_fs.popleft()
        if comment_fs["id"] == author_reply_id_to_predict_fs:
            comment_fs["score"] = "[SCORE_TO_PREDICT]"  # Hide score in stored version for FS
            break
        if "replies" in comment_fs:
            for r_fs in comment_fs["replies"]: q_fs.append(r_fs)

    if post_data_store_for_default is None: post_data_store_for_default = {}
    post_data_store_for_default[default_post_ref_id] = convert_datetimes_to_strings_in_obj(
        default_post_content_for_store)

    fs_post_summary_str = get_post_summary_string(default_post_content,
                                                  "DefaultAuthorOP")  # Use original content for summary
    fs_parent_author = "UserBeta"
    fs_parent_id_ref = f"{default_post_ref_id}-c1-r1"
    fs_parent_body_summary = "I agree. I have a follow-up question regarding the second point on data privacy."[
                             :100] + "..."
    fs_reply_author_name = "DefaultAuthorOP"
    fs_reply_id_ref = author_reply_id_to_predict_fs
    fs_reply_timestamp = "2023-01-01T13:00:00Z"
    fs_reply_body_summary = "Thanks for the question, UserBeta! That's a great point. My thoughts are that regulations need to catch up quickly."

    prompt_segment = (
        "### FEW-SHOT EXAMPLE (With Full Conversation Context) ###\n"
        f"{fs_post_summary_str}\n\n"
        f"The author of the post, '{fs_reply_author_name}', made a reply.\n"
        f"Parent Comment (that '{fs_reply_author_name}' replied to, ID: '{fs_parent_id_ref}'): Author: '{fs_parent_author}', Body Summary: \"{fs_parent_body_summary}\"\n"
        f"Author's Reply (ID: '{fs_reply_id_ref}'): Timestamp: '{fs_reply_timestamp}', Body: \"{fs_reply_body_summary}\"\n"
        f"(The actual score of this Author's Reply '{fs_reply_id_ref}' is hidden. Its full context is in post ID '{default_post_ref_id}' in the data store.)\n\n"
        f"Predict the specific numerical score for '{fs_reply_author_name}'s reply (ID: '{fs_reply_id_ref}').\n"
        "RULES FOR YOUR RESPONSE (for this example and actual task):\n"
        "- Internally decide if the score is positive or negative (do not state this step).\n"
        "- The score's absolute value MUST be 3 or greater.\n"
        "- Output ONLY the integer. NOTHING ELSE. For example: -7 or 5 or 25.\n\n"
        "Score:"
    )
    return {
        "prompt_segment": prompt_segment,
        "true_score_for_few_shot": default_fs_reply_true_score,  # Numerical score
        "fs_post_ref_id": default_post_ref_id,
        "fs_reply_id_hidden": fs_reply_id_ref  # ID of the reply whose score is hidden/predicted in the example
    }


def get_default_few_shot_for_without_prompt_refactored():  # For "WithoutConversation"
    fs_subreddit = "askhistorians"
    fs_post_id = "fs_without_post1_hist"
    fs_author = "HistoryBuff23"
    fs_title = "Impact of the printing press on Renaissance literacy"
    fs_content_summary = "Exploring how Johannes Gutenberg's invention revolutionized information dissemination and literacy rates across Europe during the Renaissance period. What were the immediate and long-term effects?"[
                         :100] + "..."
    fs_reply_author = "HistoryBuff23"  # Author replying to a comment on their own post
    fs_reply_body = "Great question! One immediate effect was the rapid spread of humanist ideas, as texts became cheaper and more accessible beyond monastic scriptoria."
    fs_true_score = 18  # Example numerical score, abs >= 3

    prompt_segment = (
        "### FEW-SHOT EXAMPLE (Without Full Conversation Context) ###\n"
        f"Consider the following post:\n"
        f"- Subreddit: r/{fs_subreddit}\n"
        f"- Post ID: {fs_post_id}\n"
        f"- Author: u/{fs_author}\n"
        f"- Title: \"{fs_title}\"\n"
        f"- Original Post Content Summary: \"{fs_content_summary}\"\n\n"
        f"The author of this post, u/{fs_reply_author}, posted a reply to a comment under their post. "
        f"The content of the author's reply is:\n"
        f"\"{fs_reply_body}\"\n\n"
        f"Predict the specific numerical score of this author's reply.\n"
        "RULES FOR YOUR RESPONSE (for this example and actual task):\n"
        "- Internally decide if the score is positive or negative (do not state this step).\n"
        "- The score's absolute value MUST be 3 or greater.\n"
        "- Output ONLY the integer. NOTHING ELSE. For example: -5 or 10 or -18.\n\n"
        "Score:"
    )
    return {
        "prompt_segment": prompt_segment,
        "true_score_for_few_shot": fs_true_score
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

    temp_post_data_store_for_default_fs_with_convo = {}
    default_few_shot_with_convo = get_default_few_shot_details_refactored(
        temp_post_data_store_for_default_fs_with_convo)
    default_few_shot_without_convo = get_default_few_shot_for_without_prompt_refactored()

    all_potential_positive_samples_to_filter = []

    user_g_idx = 0
    for main_author_name, posts_by_author_orig in all_user_data.items():
        user_g_idx += 1
        post_user_idx = 0
        for current_post_data_original_ref in posts_by_author_orig:
            post_user_idx += 1
            post_reference_id = f"u{user_g_idx}p{post_user_idx}"

            current_post_structured_with_ids_orig = deepcopy(current_post_data_original_ref)
            flat_nodes_of_current_post_orig = generate_unique_ids_for_post(
                deepcopy(current_post_structured_with_ids_orig), post_reference_id)

            author_replies_for_score_pred = find_author_replies_in_post_for_score_prediction(
                flat_nodes_of_current_post_orig, main_author_name
            )

            initial_skip = False
            participants = set(node.get('author') for node in flat_nodes_of_current_post_orig if node.get('author'))
            if len(participants) <= 3:
                initial_skip = True

            if not initial_skip:
                num_total_author_replies = sum(1 for node in flat_nodes_of_current_post_orig if
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
                if highest_score_reply['_abs_score'] <= 3:  # Keep abs score for filtering
                    initial_skip = True
                elif not highest_score_reply['_parent_body'] or \
                        clean_text(highest_score_reply['_parent_body']).lower() in ['[deleted]', '[removed]', '']:
                    initial_skip = True

            if initial_skip:
                skipped_post_count += 1
                continue

            target_reply_obj = author_replies_for_score_pred[0]
            true_numerical_score = target_reply_obj['score']

            current_post_structured_with_ids_for_store = build_dialogue_tree_with_hidden_score(
                deepcopy(current_post_structured_with_ids_orig), target_reply_obj['id']
            )
            current_post_structured_with_ids_for_store = convert_datetimes_to_strings_in_obj(
                current_post_structured_with_ids_for_store)
            current_post_structured_with_ids_no_hiding = convert_datetimes_to_strings_in_obj(
                deepcopy(current_post_structured_with_ids_orig))

            is_positive_for_filtering = target_reply_obj['score'] > 0

            # Consolidate all details needed for prompt generation
            sample_details = {
                "target_reply_obj": target_reply_obj,
                "current_post_structured_with_ids_for_store": current_post_structured_with_ids_for_store,
                "current_post_structured_with_ids_no_hiding": current_post_structured_with_ids_no_hiding,
                "post_reference_id": post_reference_id,
                "main_author_name": main_author_name,
                "author_replies_for_score_pred": author_replies_for_score_pred,
                "true_numerical_score": true_numerical_score,
                "current_subreddit": clean_text(current_post_structured_with_ids_no_hiding.get('__sub__', "N/A")),
            }

            if is_positive_for_filtering:
                all_potential_positive_samples_to_filter.append(sample_details)
            else:  # negative score samples (or zero, though abs_score <=3 is filtered) are processed directly
                # This 'sample_data' will be 'sample_details' from above
                sample_data_to_process = [sample_details]
                # Loop (of 1 item here) to use common processing logic below
                for current_sample_data in sample_data_to_process:
                    # --- "With Conversation" Prompt ---
                    post_data_store_with_convo = {}
                    post_data_store_with_convo[current_sample_data["post_reference_id"]] = current_sample_data[
                        "current_post_structured_with_ids_for_store"]

                    second_highest_score_reply_for_fs = None
                    if len(current_sample_data["author_replies_for_score_pred"]) > 1:
                        temp_second_reply = current_sample_data["author_replies_for_score_pred"][1]
                        if isinstance(temp_second_reply.get('score'), (int, float)) and \
                                temp_second_reply.get('body') and clean_text(
                            temp_second_reply.get('body')).lower() not in ['', '[deleted]', '[removed]'] and \
                                temp_second_reply.get('_parent_body') and clean_text(
                            temp_second_reply.get('_parent_body')).lower() not in ['', '[deleted]', '[removed]']:
                            second_highest_score_reply_for_fs = temp_second_reply

                    fs_prompt_segment_with = default_few_shot_with_convo["prompt_segment"]
                    fs_true_answer_with = default_few_shot_with_convo["true_score_for_few_shot"]
                    if default_few_shot_with_convo["fs_post_ref_id"] not in post_data_store_with_convo:
                        post_data_store_with_convo[default_few_shot_with_convo["fs_post_ref_id"]] = \
                        temp_post_data_store_for_default_fs_with_convo[default_few_shot_with_convo["fs_post_ref_id"]]

                    fs_target_reply_local_with = None
                    if second_highest_score_reply_for_fs:
                        fs_target_reply_local_with = second_highest_score_reply_for_fs
                        fs_post_summary_str = get_post_summary_string(
                            current_sample_data["current_post_structured_with_ids_no_hiding"],
                            current_sample_data["main_author_name"])
                        fs_parent_author = fs_target_reply_local_with['_parent_author']
                        fs_parent_id_ref = fs_target_reply_local_with['_parent_id']
                        fs_parent_body_summary = clean_text(fs_target_reply_local_with['_parent_body'])[:100] + "..."
                        fs_reply_author_name = current_sample_data["main_author_name"]
                        fs_reply_id_ref = fs_target_reply_local_with['id']
                        fs_reply_timestamp = datetime_to_string(fs_target_reply_local_with['_parsed_timestamp'])
                        fs_reply_body_summary = clean_text(fs_target_reply_local_with['body'])
                        fs_true_numerical_score_for_fs_example = fs_target_reply_local_with['score']
                        fs_prompt_segment_with = (
                            "### FEW-SHOT EXAMPLE (With Full Conversation Context) ###\n"
                            f"{fs_post_summary_str}\n\n"
                            f"The author of the post, '{fs_reply_author_name}', made a reply.\n"
                            f"Parent Comment (that '{fs_reply_author_name}' replied to, ID: '{fs_parent_id_ref}'): Author: '{fs_parent_author}', Body Summary: \"{fs_parent_body_summary}\"\n"
                            f"Author's Reply (ID: '{fs_reply_id_ref}'): Timestamp: '{fs_reply_timestamp}', Body: \"{fs_reply_body_summary}\"\n"
                            f"(The actual score of this Author's Reply '{fs_reply_id_ref}' is hidden. Its full context is in post ID '{current_sample_data['post_reference_id']}' in the data store.)\n\n"
                            f"Predict the specific numerical score for '{fs_reply_author_name}'s reply (ID: '{fs_reply_id_ref}').\n"
                            "RULES FOR YOUR RESPONSE (for this example and actual task):\n"
                            "- Internally decide if the score is positive or negative (do not state this step).\n"
                            "- The score's absolute value MUST be 3 or greater.\n"
                            "- Output ONLY the integer. NOTHING ELSE. For example: -7 or 5 or 25.\n\n"
                            "Score:"
                        )
                        fs_true_answer_with = fs_true_numerical_score_for_fs_example

                    main_post_summary_str = get_post_summary_string(
                        current_sample_data["current_post_structured_with_ids_no_hiding"],
                        current_sample_data["main_author_name"])
                    main_parent_author = current_sample_data["target_reply_obj"]['_parent_author']
                    main_parent_id_ref = current_sample_data["target_reply_obj"]['_parent_id']
                    main_parent_body_summary = clean_text(current_sample_data["target_reply_obj"]['_parent_body'])[
                                               :100] + "..."
                    main_target_reply_author_name = current_sample_data["main_author_name"]
                    main_target_reply_id_ref = current_sample_data["target_reply_obj"]['id']
                    main_target_reply_timestamp = datetime_to_string(
                        current_sample_data["target_reply_obj"]['_parsed_timestamp'])
                    main_target_reply_body_summary = clean_text(current_sample_data["target_reply_obj"]['body'])

                    sign_hint_actual_task = ""
                    if PROMPT_SIGN_HINT:
                        if current_sample_data["true_numerical_score"] > 0:
                            sign_hint_actual_task = " (Hint: the score is expected to be positive)."
                        elif current_sample_data["true_numerical_score"] < 0:
                            sign_hint_actual_task = " (Hint: the score is expected to be negative)."

                    prompt_text_for_with_convo_task = (
                        f"{fs_prompt_segment_with}\n{fs_true_answer_with}\n\n"
                        "### ACTUAL TASK (With Full Conversation Context) ###\n"
                        f"{main_post_summary_str}\n\n"
                        f"The author of the post, '{main_target_reply_author_name}', made another reply (or the primary reply we are interested in).\n"
                        f"Parent Comment (that '{main_target_reply_author_name}' replied to, ID: '{main_parent_id_ref}'): Author: '{main_parent_author}', Body Summary: \"{main_parent_body_summary}\"\n"
                        f"Author's Reply (ID: '{main_target_reply_id_ref}'): Timestamp: '{main_target_reply_timestamp}', Body: \"{main_target_reply_body_summary}\"\n"
                        f"(Your task is to predict the specific numerical score of this Author's Reply '{main_target_reply_id_ref}'. Its score is hidden as '[SCORE_TO_PREDICT]' in its full context within post ID '{current_sample_data['post_reference_id']}' in the data store.){sign_hint_actual_task}\n\n"
                        f"Predict the specific numerical score for '{main_target_reply_author_name}'s reply (ID: '{main_target_reply_id_ref}').\n"
                        "RULES FOR YOUR RESPONSE:\n"
                        "- Internally decide if the score is positive or negative (do not state this step).\n"
                        "- The score's absolute value MUST be 3 or greater.\n"
                        "- Output ONLY the integer. NOTHING ELSE. For example: -7 or 5 or 25.\n\n"
                        "Score:"
                    )
                    output_data_lines_with_convo.append({
                        "post_data_store": post_data_store_with_convo,
                        "prediction_task": {
                            "prompt_text": prompt_text_for_with_convo_task,
                            "target_post_reference_id": current_sample_data["post_reference_id"],
                            "target_reply_id": main_target_reply_id_ref,
                            "true_label": current_sample_data["true_numerical_score"],
                            "subreddit": current_sample_data["current_subreddit"],
                            "few_shot_info": {
                                "type": "dynamic" if second_highest_score_reply_for_fs else "default",
                                "fs_post_ref_id": current_sample_data[
                                    "post_reference_id"] if second_highest_score_reply_for_fs else
                                default_few_shot_with_convo["fs_post_ref_id"],
                                "fs_reply_id_hidden_in_example": fs_target_reply_local_with[
                                    'id'] if second_highest_score_reply_for_fs and fs_target_reply_local_with else
                                default_few_shot_with_convo["fs_reply_id_hidden"]
                            }
                        }
                    })

                    # --- "Without Conversation" Prompt ---
                    post_data_store_without_convo = {}
                    simplified_post_for_store = {k: v for k, v in current_sample_data[
                        "current_post_structured_with_ids_no_hiding"].items() if k not in ['comments', 'replies']}
                    post_data_store_without_convo[current_sample_data["post_reference_id"]] = simplified_post_for_store

                    fs_prompt_without = default_few_shot_without_convo["prompt_segment"]
                    fs_answer_without = default_few_shot_without_convo["true_score_for_few_shot"]

                    without_convo_prompt_text = (
                        f"{fs_prompt_without}\n{fs_answer_without}\n\n"
                        "### ACTUAL TASK (Without Full Conversation Context) ###\n"
                        f"Consider the following post:\n"
                        f"- Subreddit: r/{current_sample_data['current_subreddit']}\n"
                        f"- Post ID: {current_sample_data['post_reference_id']}\n"
                        f"- Author: u/{current_sample_data['main_author_name']}\n"
                        f"- Title: \"{clean_text(current_sample_data['current_post_structured_with_ids_no_hiding'].get('title', 'N/A'))}\"\n"
                        f"- Original Post Content Summary: \"{clean_text(current_sample_data['current_post_structured_with_ids_no_hiding'].get('content', 'N/A'))[:500] + ('...' if len(clean_text(current_sample_data['current_post_structured_with_ids_no_hiding'].get('content', 'N/A'))) > 500 else '')}\"\n\n"
                        f"The author of this post, u/{current_sample_data['main_author_name']}, posted a reply to a comment under their post. "
                        f"The content of the author's reply is:\n"
                        f"\"{clean_text(current_sample_data['target_reply_obj']['body'])}\"\n\n"
                        f"Predict the specific numerical score of this author's reply.{sign_hint_actual_task}\n"
                        "RULES FOR YOUR RESPONSE:\n"
                        "- Internally decide if the score is positive or negative (do not state this step).\n"
                        "- The score's absolute value MUST be 3 or greater.\n"
                        "- Output ONLY the integer. NOTHING ELSE. For example: -3 or 8 or -12.\n\n"
                        "Score:"
                    )
                    output_data_lines_without_convo.append({
                        "post_data_store": post_data_store_without_convo,
                        "prediction_task": {
                            "prompt_text": without_convo_prompt_text,
                            "target_post_reference_id": current_sample_data["post_reference_id"],
                            "target_reply_id": main_target_reply_id_ref,
                            "true_label": current_sample_data["true_numerical_score"],
                            "subreddit": current_sample_data["current_subreddit"],
                        }
                    })
                    processed_post_count += 1

    # Filter positive samples (those with score > 0)
    all_potential_positive_samples_to_filter.sort(key=lambda x: x["target_reply_obj"]["_abs_score"])
    num_to_delete = len(all_potential_positive_samples_to_filter) // 2
    samples_to_keep = all_potential_positive_samples_to_filter[num_to_delete:]
    deleted_half_positive_low_score_count = num_to_delete

    for current_sample_data in samples_to_keep:  # These are positive score samples kept after filtering
        # --- "With Conversation" Prompt ---
        post_data_store_with_convo = {}
        post_data_store_with_convo[current_sample_data["post_reference_id"]] = current_sample_data[
            "current_post_structured_with_ids_for_store"]

        second_highest_score_reply_for_fs = None
        if len(current_sample_data["author_replies_for_score_pred"]) > 1:
            temp_second_reply = current_sample_data["author_replies_for_score_pred"][1]
            if isinstance(temp_second_reply.get('score'), (int, float)) and \
                    temp_second_reply.get('body') and clean_text(temp_second_reply.get('body')).lower() not in ['',
                                                                                                                '[deleted]',
                                                                                                                '[removed]'] and \
                    temp_second_reply.get('_parent_body') and clean_text(
                temp_second_reply.get('_parent_body')).lower() not in ['', '[deleted]', '[removed]']:
                second_highest_score_reply_for_fs = temp_second_reply

        fs_prompt_segment_with = default_few_shot_with_convo["prompt_segment"]
        fs_true_answer_with = default_few_shot_with_convo["true_score_for_few_shot"]
        if default_few_shot_with_convo["fs_post_ref_id"] not in post_data_store_with_convo:
            post_data_store_with_convo[default_few_shot_with_convo["fs_post_ref_id"]] = \
            temp_post_data_store_for_default_fs_with_convo[default_few_shot_with_convo["fs_post_ref_id"]]

        fs_target_reply_local_with = None
        if second_highest_score_reply_for_fs:
            fs_target_reply_local_with = second_highest_score_reply_for_fs
            fs_post_summary_str = get_post_summary_string(
                current_sample_data["current_post_structured_with_ids_no_hiding"],
                current_sample_data["main_author_name"])
            fs_parent_author = fs_target_reply_local_with['_parent_author']
            fs_parent_id_ref = fs_target_reply_local_with['_parent_id']
            fs_parent_body_summary = clean_text(fs_target_reply_local_with['_parent_body'])[:100] + "..."
            fs_reply_author_name = current_sample_data["main_author_name"]
            fs_reply_id_ref = fs_target_reply_local_with['id']
            fs_reply_timestamp = datetime_to_string(fs_target_reply_local_with['_parsed_timestamp'])
            fs_reply_body_summary = clean_text(fs_target_reply_local_with['body'])
            fs_true_numerical_score_for_fs_example = fs_target_reply_local_with['score']
            fs_prompt_segment_with = (
                "### FEW-SHOT EXAMPLE (With Full Conversation Context) ###\n"
                f"{fs_post_summary_str}\n\n"
                f"The author of the post, '{fs_reply_author_name}', made a reply.\n"
                f"Parent Comment (that '{fs_reply_author_name}' replied to, ID: '{fs_parent_id_ref}'): Author: '{fs_parent_author}', Body Summary: \"{fs_parent_body_summary}\"\n"
                f"Author's Reply (ID: '{fs_reply_id_ref}'): Timestamp: '{fs_reply_timestamp}', Body: \"{fs_reply_body_summary}\"\n"
                f"(The actual score of this Author's Reply '{fs_reply_id_ref}' is hidden. Its full context is in post ID '{current_sample_data['post_reference_id']}' in the data store.)\n\n"
                f"Predict the specific numerical score for '{fs_reply_author_name}'s reply (ID: '{fs_reply_id_ref}').\n"
                "RULES FOR YOUR RESPONSE (for this example and actual task):\n"
                "- Internally decide if the score is positive or negative (do not state this step).\n"
                "- The score's absolute value MUST be 3 or greater.\n"
                "- Output ONLY the integer. NOTHING ELSE. For example: -7 or 5 or 25.\n\n"
                "Score:"
            )
            fs_true_answer_with = fs_true_numerical_score_for_fs_example

        main_post_summary_str = get_post_summary_string(
            current_sample_data["current_post_structured_with_ids_no_hiding"], current_sample_data["main_author_name"])
        main_parent_author = current_sample_data["target_reply_obj"]['_parent_author']
        main_parent_id_ref = current_sample_data["target_reply_obj"]['_parent_id']
        main_parent_body_summary = clean_text(current_sample_data["target_reply_obj"]['_parent_body'])[:100] + "..."
        main_target_reply_author_name = current_sample_data["main_author_name"]
        main_target_reply_id_ref = current_sample_data["target_reply_obj"]['id']
        main_target_reply_timestamp = datetime_to_string(current_sample_data["target_reply_obj"]['_parsed_timestamp'])
        main_target_reply_body_summary = clean_text(current_sample_data["target_reply_obj"]['body'])

        sign_hint_actual_task = ""
        if PROMPT_SIGN_HINT:
            if current_sample_data["true_numerical_score"] > 0:
                sign_hint_actual_task = " (Hint: the score is expected to be positive)."
            elif current_sample_data["true_numerical_score"] < 0:
                sign_hint_actual_task = " (Hint: the score is expected to be negative)."

        prompt_text_for_with_convo_task = (
            f"{fs_prompt_segment_with}\n{fs_true_answer_with}\n\n"
            "### ACTUAL TASK (With Full Conversation Context) ###\n"
            f"{main_post_summary_str}\n\n"
            f"The author of the post, '{main_target_reply_author_name}', made another reply (or the primary reply we are interested in).\n"
            f"Parent Comment (that '{main_target_reply_author_name}' replied to, ID: '{main_parent_id_ref}'): Author: '{main_parent_author}', Body Summary: \"{main_parent_body_summary}\"\n"
            f"Author's Reply (ID: '{main_target_reply_id_ref}'): Timestamp: '{main_target_reply_timestamp}', Body: \"{main_target_reply_body_summary}\"\n"
            f"(Your task is to predict the specific numerical score of this Author's Reply '{main_target_reply_id_ref}'. Its score is hidden as '[SCORE_TO_PREDICT]' in its full context within post ID '{current_sample_data['post_reference_id']}' in the data store.){sign_hint_actual_task}\n\n"
            f"Predict the specific numerical score for '{main_target_reply_author_name}'s reply (ID: '{main_target_reply_id_ref}').\n"
            "RULES FOR YOUR RESPONSE:\n"
            "- Internally decide if the score is positive or negative (do not state this step).\n"
            "- The score's absolute value MUST be 3 or greater.\n"
            "- Output ONLY the integer. NOTHING ELSE. For example: -7 or 5 or 25.\n\n"
            "Score:"
        )
        output_data_lines_with_convo.append({
            "post_data_store": post_data_store_with_convo,
            "prediction_task": {
                "prompt_text": prompt_text_for_with_convo_task,
                "target_post_reference_id": current_sample_data["post_reference_id"],
                "target_reply_id": main_target_reply_id_ref,
                "true_label": current_sample_data["true_numerical_score"],
                "subreddit": current_sample_data["current_subreddit"],
                "few_shot_info": {
                    "type": "dynamic" if second_highest_score_reply_for_fs else "default",
                    "fs_post_ref_id": current_sample_data["post_reference_id"] if second_highest_score_reply_for_fs else
                    default_few_shot_with_convo["fs_post_ref_id"],
                    "fs_reply_id_hidden_in_example": fs_target_reply_local_with[
                        'id'] if second_highest_score_reply_for_fs and fs_target_reply_local_with else
                    default_few_shot_with_convo["fs_reply_id_hidden"]
                }
            }
        })

        # --- "Without Conversation" Prompt (for positive samples) ---
        post_data_store_without_convo = {}
        simplified_post_for_store = {k: v for k, v in
                                     current_sample_data["current_post_structured_with_ids_no_hiding"].items() if
                                     k not in ['comments', 'replies']}
        post_data_store_without_convo[current_sample_data["post_reference_id"]] = simplified_post_for_store

        fs_prompt_without = default_few_shot_without_convo["prompt_segment"]
        fs_answer_without = default_few_shot_without_convo["true_score_for_few_shot"]

        without_convo_prompt_text = (
            f"{fs_prompt_without}\n{fs_answer_without}\n\n"
            "### ACTUAL TASK (Without Full Conversation Context) ###\n"
            f"Consider the following post:\n"
            f"- Subreddit: r/{current_sample_data['current_subreddit']}\n"
            f"- Post ID: {current_sample_data['post_reference_id']}\n"
            f"- Author: u/{current_sample_data['main_author_name']}\n"
            f"- Title: \"{clean_text(current_sample_data['current_post_structured_with_ids_no_hiding'].get('title', 'N/A'))}\"\n"
            f"- Original Post Content Summary: \"{clean_text(current_sample_data['current_post_structured_with_ids_no_hiding'].get('content', 'N/A'))[:500] + ('...' if len(clean_text(current_sample_data['current_post_structured_with_ids_no_hiding'].get('content', 'N/A'))) > 500 else '')}\"\n\n"
            f"The author of this post, u/{current_sample_data['main_author_name']}, posted a reply to a comment under their post. "
            f"The content of the author's reply is:\n"
            f"\"{clean_text(current_sample_data['target_reply_obj']['body'])}\"\n\n"
            f"Predict the specific numerical score of this author's reply.{sign_hint_actual_task}\n"
            "RULES FOR YOUR RESPONSE:\n"
            "- Internally decide if the score is positive or negative (do not state this step).\n"
            "- The score's absolute value MUST be 3 or greater.\n"
            "- Output ONLY the integer. NOTHING ELSE. For example: -3 or 8 or -12.\n\n"
            "Score:"
        )
        output_data_lines_without_convo.append({
            "post_data_store": post_data_store_without_convo,
            "prediction_task": {
                "prompt_text": without_convo_prompt_text,
                "target_post_reference_id": current_sample_data["post_reference_id"],
                "target_reply_id": main_target_reply_id_ref,
                "true_label": current_sample_data["true_numerical_score"],
                "subreddit": current_sample_data["current_subreddit"],
            }
        })
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
    print(f"Skipped {skipped_post_count} posts due to initial filtering criteria.")
    print(
        f"Additionally, {deleted_half_positive_low_score_count} positive low-score samples were filtered out before final processing.")
    # Note: skipped_by_probability_count was not used in the provided script logic, so it remains 0 unless used elsewhere.


if __name__ == '__main__':
    if not os.path.exists(INPUT_JSON_FILE):
        print(f"CRITICAL ERROR: Input file '{INPUT_JSON_FILE}' not found. Please create it or check the path.")
        sys.exit(1)
    main()