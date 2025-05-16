import json
import os
import sys
from datetime import datetime, timezone
from copy import deepcopy
from collections import deque
import random

# --- Configuration ---
INPUT_JSON_FILE = "merged_posts_remapped.json"
# Output will be a single JSONL file where each line is an object
# containing the post_data_store and the prediction_task.
OUTPUT_JSONL_FILE = "WithConversationPrompts_PseudoRandomContext_ExactScorePrediction.jsonl"  # 文件名更改以反映新逻辑
# WITHOUT_CONVO_OUTPUT_JSONL is no longer the focus.
# WITHOUT_CONVO_OUTPUT_JSONL = "WithoutConversationPrompts_ExactScorePrediction.jsonl"

PROMPT_SIGN_HINT = True  # 用户定义的提示分数符号标志

skipped_post_count = 0
processed_post_count = 0
deleted_half_positive_low_score_count = 0


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
        try:
            dt_obj = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        except ValueError:
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
            elif k.startswith('_') and k not in ['__sub__']:
                continue
            else:
                new_dict[k] = convert_datetimes_to_strings_in_obj(v)
        return new_dict
    return obj


def generate_unique_ids_for_post(post_data, post_index_prefix):
    # This function remains largely the same as it's for structuring individual posts
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
    # This function remains largely the same for identifying suitable replies
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
    # This function is still needed for the actual target post's data store entry
    context_data = deepcopy(post_data_with_ids)
    q = deque()
    # Post scores are not hidden for this task, only reply scores
    if 'comments' in context_data and isinstance(context_data.get('comments'), list):
        for comment in context_data['comments']:
            if isinstance(comment, dict): q.append(comment)
    while q:
        node = q.popleft()
        if not isinstance(node, dict): continue
        if node.get('id') == node_id_to_hide_score_of:
            if 'score' in node:
                node['score'] = "[SCORE_TO_PREDICT]"
        if 'replies' in node and isinstance(node['replies'], list):
            for reply in node['replies']:
                if isinstance(reply, dict): q.append(reply)
    return context_data


def get_context_post_summary_string(post_obj, main_author_name_for_summary="the post author"):
    # Used for summarizing the "context provider" post (the one 200 ahead)
    title = clean_text(post_obj.get('title', "N/A"))
    subreddit = clean_text(post_obj.get('__sub__', "N/A"))
    author = clean_text(post_obj.get('author', "N/A"))
    score = post_obj.get('score', "N/A")
    timestamp = datetime_to_string(parse_timestamp(post_obj.get('timestamp'))) or "N/A"
    content_full = clean_text(post_obj.get('content', post_obj.get('body', "N/A")))
    content_summary = content_full[:300] + ('...' if len(content_full) > 300 else '')  # Shorter summary for context
    post_id = post_obj.get('id', "N/A")

    # Note: This summary is for a *different* post than the one being predicted.
    # We are explaining that this context is somewhat "randomized".
    return (
        f"The following is an unrelated post (ID: {post_id}) provided as a pseudo-random contextual example:\n"
        f"- Subreddit: r/{subreddit}\n"
        f"- Author: u/{author}\n"
        f"- Score: {score}\n"
        f"- Timestamp: {timestamp}\n"
        f"- Title: \"{title}\"\n"
        f"- Content Summary: \"{content_summary}\"\n"
        f"(This post is distinct from the actual task's target post and is used to study the effect of varied contextual prompts. "
        f"Its full dialogue tree might be available in the data store under its own ID: '{post_id}' if it's used for a few-shot example.)"
    )


def get_pseudo_random_few_shot_details(context_provider_sample_details, post_data_store_for_fs_context):
    # This function will now use the context_provider_sample_details
    # to construct a few-shot example.
    # The context_provider_sample_details contains:
    # "target_reply_obj", "current_post_structured_with_ids_no_hiding",
    # "post_reference_id", "main_author_name", "true_numerical_score" (of the context reply)

    fs_context_post_full = context_provider_sample_details["current_post_structured_with_ids_no_hiding"]
    fs_context_reply_obj = context_provider_sample_details[
        "target_reply_obj"]  # This is a reply from the *context provider* post
    fs_context_post_ref_id = context_provider_sample_details["post_reference_id"]
    fs_context_post_author = context_provider_sample_details["main_author_name"]  # Author of the context provider post
    fs_true_score_for_this_example = context_provider_sample_details[
        "true_numerical_score"]  # Score of the reply in context provider post

    # Store the context provider's post (with its example reply's score hidden)
    # if it's not already there (e.g. if it's the same as the default one, which is unlikely now)
    if fs_context_post_ref_id not in post_data_store_for_fs_context:
        # Create a version where this *example* reply's score is hidden for the FS prompt
        fs_context_post_for_store = build_dialogue_tree_with_hidden_score(
            deepcopy(fs_context_post_full), fs_context_reply_obj['id']
        )
        post_data_store_for_fs_context[fs_context_post_ref_id] = convert_datetimes_to_strings_in_obj(
            fs_context_post_for_store)

    fs_post_summary_str = get_context_post_summary_string(fs_context_post_full, fs_context_post_author)

    fs_parent_author = fs_context_reply_obj['_parent_author']
    fs_parent_id_ref = fs_context_reply_obj['_parent_id']
    fs_parent_body_summary = clean_text(fs_context_reply_obj['_parent_body'])[:100] + "..."
    # fs_reply_author_name is the author of the context post, who made the example reply
    fs_reply_author_name = fs_context_post_author
    fs_reply_id_ref = fs_context_reply_obj['id']
    fs_reply_timestamp = datetime_to_string(fs_context_reply_obj['_parsed_timestamp'])
    fs_reply_body_summary = clean_text(fs_context_reply_obj['body'])

    # Modified introductory text for the few-shot example.
    prompt_intro_text = (
        "This is a study on predicting Reddit reply scores. The context for few-shot examples and tasks "
        "is intentionally drawn from posts that are distant (e.g., 200 posts away) from the "
        "primary post whose reply score you need to predict. This is to test the robustness of predictions "
        "with somewhat randomized contextual information.\n\n"
    )

    prompt_segment = (
        f"{prompt_intro_text}"
        "### FEW-SHOT EXAMPLE (Using Pseudo-Random Context from a Distant Post) ###\n"
        f"{fs_post_summary_str}\n\n"  # This summary already explains it's an unrelated post
        f"Within this distant post, its author, '{fs_reply_author_name}', made a reply.\n"
        f"Parent Comment (that '{fs_reply_author_name}' replied to in this distant post, ID: '{fs_parent_id_ref}'):\n"
        f"  - Author: '{fs_parent_author}'\n"
        f"  - Body Summary: \"{fs_parent_body_summary}\"\n"
        f"  - Score: {fs_context_reply_obj['_parent_score']}\n"  # Added parent score
        f"  - Timestamp: {datetime_to_string(parse_timestamp(fs_context_reply_obj['_parent_timestamp']))}\n\n"  # Added parent timestamp
        f"Author's Reply in this distant post (ID: '{fs_reply_id_ref}'):\n"
        f"  - Timestamp: '{fs_reply_timestamp}'\n"
        f"  - Body: \"{fs_reply_body_summary}\"\n"
        f"(The actual score of this reply '{fs_reply_id_ref}' from the distant post is hidden for this example. "
        f"Its full context is in the distant post ID '{fs_context_post_ref_id}' in the data store.)\n\n"
        f"Predict the specific numerical score for '{fs_reply_author_name}'s reply (ID: '{fs_reply_id_ref}') within this distant post example.\n"
        "RULES FOR YOUR RESPONSE (for this example and actual task):\n"
        "- Your goal is to predict the integer score.\n"
        "- The score's absolute value MUST be 3 or greater.\n"
        "- Output ONLY the integer. NOTHING ELSE. For example: -7 or 5 or 25.\n\n"
        "Score:"
    )
    return {
        "prompt_segment": prompt_segment,
        "true_score_for_few_shot": fs_true_score_for_this_example,
        "fs_post_ref_id": fs_context_post_ref_id,  # Ref to the distant post used for FS
        "fs_reply_id_hidden": fs_reply_id_ref
    }


# --- Main Processing ---
def main():
    global skipped_post_count, processed_post_count, deleted_half_positive_low_score_count

    if not os.path.exists(INPUT_JSON_FILE):
        print(f"Error: Input file '{INPUT_JSON_FILE}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
        all_user_data = json.load(f)

    output_data_lines_with_convo = []

    # First pass: Collect all valid samples that can be targets for prediction
    # AND can serve as context providers.
    all_valid_samples_for_tasks_and_context = []
    user_g_idx_collect = 0
    for main_author_name_collect, posts_by_author_orig_collect in all_user_data.items():
        user_g_idx_collect += 1
        post_user_idx_collect = 0
        for current_post_data_original_ref_collect in posts_by_author_orig_collect:
            post_user_idx_collect += 1
            post_reference_id_collect = f"u{user_g_idx_collect}p{post_user_idx_collect}"

            current_post_structured_with_ids_orig_collect = deepcopy(current_post_data_original_ref_collect)
            flat_nodes_of_current_post_orig_collect = generate_unique_ids_for_post(
                deepcopy(current_post_structured_with_ids_orig_collect), post_reference_id_collect)

            author_replies_for_score_pred_collect = find_author_replies_in_post_for_score_prediction(
                flat_nodes_of_current_post_orig_collect, main_author_name_collect
            )

            initial_skip_collect = False
            participants_collect = set(
                node.get('author') for node in flat_nodes_of_current_post_orig_collect if node.get('author'))
            if len(participants_collect) <= 3:
                initial_skip_collect = True

            if not initial_skip_collect:
                num_total_author_replies_collect = sum(1 for node in flat_nodes_of_current_post_orig_collect if
                                                       node.get('type') == 'comment' and
                                                       node.get('author') == main_author_name_collect and
                                                       node.get('body') and
                                                       clean_text(node.get('body')).lower() not in ['', '[deleted]',
                                                                                                    '[removed]'])
                if num_total_author_replies_collect <= 2:  # At least one for target, one for potential FS from same post (though FS now from +200)
                    initial_skip_collect = True

            if not initial_skip_collect and not author_replies_for_score_pred_collect:
                initial_skip_collect = True

            if not initial_skip_collect:
                # Check if the highest scoring reply is valid
                highest_score_reply_collect = author_replies_for_score_pred_collect[0]
                if highest_score_reply_collect['_abs_score'] <= 3:
                    initial_skip_collect = True
                elif not highest_score_reply_collect['_parent_body'] or \
                        clean_text(highest_score_reply_collect['_parent_body']).lower() in ['[deleted]', '[removed]',
                                                                                            '']:
                    initial_skip_collect = True

            if initial_skip_collect:
                skipped_post_count += 1  # Count here as it's a skip for being a *target*
                continue

            # If we reach here, the post is valid to be a *target* for prediction
            # and its highest scoring reply can be the one predicted.
            # It can also serve as a *context provider* for another post's prediction task.
            target_reply_obj_collect = author_replies_for_score_pred_collect[0]

            sample_details_collect = {
                "target_reply_obj": target_reply_obj_collect,
                # The reply whose score might be predicted OR used in a FS example
                "current_post_structured_with_ids_no_hiding": convert_datetimes_to_strings_in_obj(
                    deepcopy(current_post_structured_with_ids_orig_collect)),
                "post_reference_id": post_reference_id_collect,
                "main_author_name": main_author_name_collect,
                "true_numerical_score": target_reply_obj_collect['score'],  # Score of the target_reply_obj
                "current_subreddit": clean_text(current_post_structured_with_ids_orig_collect.get('__sub__', "N/A")),
                "is_positive_for_filtering": target_reply_obj_collect['score'] > 0,
                # Store the original full structure with IDs for later use if this post becomes a context provider
                "_original_full_post_structure_with_ids": deepcopy(current_post_structured_with_ids_orig_collect),
                "_author_replies_for_score_pred_list": deepcopy(author_replies_for_score_pred_collect)
                # For dynamic FS from *context* post
            }
            all_valid_samples_for_tasks_and_context.append(sample_details_collect)

    # Filter positive samples based on score (same logic as before, but on the collected list)
    neg_samples = [s for s in all_valid_samples_for_tasks_and_context if not s["is_positive_for_filtering"]]
    pos_samples = [s for s in all_valid_samples_for_tasks_and_context if s["is_positive_for_filtering"]]

    pos_samples.sort(key=lambda x: x["target_reply_obj"]["_abs_score"])  # Sort by absolute score
    num_pos_to_delete = len(pos_samples) // 2
    kept_pos_samples = pos_samples[num_pos_to_delete:]
    deleted_half_positive_low_score_count = num_pos_to_delete

    final_samples_to_process = neg_samples + kept_pos_samples
    random.shuffle(final_samples_to_process)  # Shuffle to mix positive/negative for processing order

    if not final_samples_to_process:
        print("No valid samples found after initial collection and filtering. Exiting.", file=sys.stderr)
        sys.exit(1)

    num_total_samples = len(final_samples_to_process)

    # Second pass: Generate prompts using the +200 context logic
    for i, current_sample_data in enumerate(final_samples_to_process):

        # --- Determine the Context Provider Sample ---
        # This is the sample that is 200 posts *after* the current_sample_data in the `final_samples_to_process` list
        context_provider_index = (i + 200) % num_total_samples
        context_provider_sample_details = final_samples_to_process[context_provider_index]

        # --- "With Conversation" Prompt (Pseudo-Random Context) ---
        post_data_store_with_convo = {}

        # 1. Store the *actual target* post's data (with its target reply score hidden)
        #    The `current_sample_data` is the one whose reply we are predicting.
        target_post_ref_id = current_sample_data["post_reference_id"]
        target_reply_to_predict_obj = current_sample_data["target_reply_obj"]

        # Use the _original_full_post_structure_with_ids from current_sample_data
        # to build the version with the target reply score hidden.
        target_post_structured_for_store = build_dialogue_tree_with_hidden_score(
            deepcopy(current_sample_data["_original_full_post_structure_with_ids"]),
            # Use the preserved original structure
            target_reply_to_predict_obj['id']
        )
        post_data_store_with_convo[target_post_ref_id] = convert_datetimes_to_strings_in_obj(
            target_post_structured_for_store)

        # 2. Prepare Few-Shot example using the `context_provider_sample_details`
        #    The `post_data_store_with_convo` will also store the context provider's post if needed for FS.
        fs_details = get_pseudo_random_few_shot_details(context_provider_sample_details, post_data_store_with_convo)
        fs_prompt_segment_with = fs_details["prompt_segment"]
        fs_true_answer_with = fs_details["true_score_for_few_shot"]
        # The fs_details function already added the context provider's post to post_data_store_with_convo if necessary.

        # 3. Prepare the ACTUAL TASK section using `current_sample_data`
        #    The summary string for the *actual task* should refer to the *actual target post*.
        #    However, the instructions will clarify that the overall context (including FS) is from a distant post.
        actual_task_post_summary_str = (
            f"Referenced Post for ACTUAL TASK (ID: {target_post_ref_id}):\n"
            f"- Subreddit: r/{current_sample_data['current_subreddit']}\n"
            f"- Author: u/{current_sample_data['main_author_name']}\n"
            f"- Score: {current_sample_data['current_post_structured_with_ids_no_hiding'].get('score', 'N/A')}\n"  # Score of the target post
            f"- Timestamp: {current_sample_data['current_post_structured_with_ids_no_hiding'].get('timestamp', 'N/A')}\n"
            f"- Title: \"{clean_text(current_sample_data['current_post_structured_with_ids_no_hiding'].get('title', 'N/A'))}\"\n"
            f"- Content Summary: \"{clean_text(current_sample_data['current_post_structured_with_ids_no_hiding'].get('content', 'N/A'))[:300] + '...'}\"\n"
            f"(Full post content and dialogue tree for THIS actual task post are available in the data store under post reference ID '{target_post_ref_id}')"
        )

        main_parent_author = target_reply_to_predict_obj['_parent_author']
        main_parent_id_ref = target_reply_to_predict_obj['_parent_id']
        main_parent_body_full = clean_text(target_reply_to_predict_obj['_parent_body'])  # Full body for task prompt
        main_parent_score = target_reply_to_predict_obj['_parent_score']  # Parent score for task prompt
        main_parent_timestamp = datetime_to_string(
            parse_timestamp(target_reply_to_predict_obj['_parent_timestamp']))  # Parent timestamp

        main_target_reply_author_name = current_sample_data["main_author_name"]  # Author of the reply to predict
        main_target_reply_id_ref = target_reply_to_predict_obj['id']
        main_target_reply_timestamp = datetime_to_string(target_reply_to_predict_obj['_parsed_timestamp'])
        main_target_reply_body_full = clean_text(target_reply_to_predict_obj['body'])  # Full body for task prompt

        true_numerical_score_for_task = current_sample_data["true_numerical_score"]

        sign_hint_actual_task = ""
        if PROMPT_SIGN_HINT:
            if true_numerical_score_for_task > 0:
                sign_hint_actual_task = " (Hint: the score is expected to be positive)."
            elif true_numerical_score_for_task < 0:
                sign_hint_actual_task = " (Hint: the score is expected to be negative)."

        # Natural language description for the prediction task
        prediction_request_text = (
            f"Your task is to predict the specific numerical score for '{main_target_reply_author_name}'s reply (ID: '{main_target_reply_id_ref}') "
            f"within the ACTUAL TASK post (ID: '{target_post_ref_id}'). "
            f"The reply's score is hidden as '[SCORE_TO_PREDICT]' in its full context within post ID '{target_post_ref_id}' in the data store."
            f"{sign_hint_actual_task}\n\n"
            "Please provide ONLY the integer score based on the information given for the ACTUAL TASK reply and its direct parent, "
            "keeping in mind the rules from the pseudo-random context few-shot example."
        )

        prompt_text_for_with_convo_task = (
            f"{fs_prompt_segment_with}\n{fs_true_answer_with}\n\n"
            "### ACTUAL TASK (Predict Score for a Reply in a *Different* Post) ###\n"
            # Reiterate the context shift for clarity
            "Reminder: The few-shot example above used a pseudo-random distant post for context. "
            "Now, you will predict the score for a reply in a *new, specific* target post detailed below.\n\n"
            f"{actual_task_post_summary_str}\n\n"
            f"Within this ACTUAL TASK post (ID: '{target_post_ref_id}'), the post's author, '{main_target_reply_author_name}', made the reply you need to score.\n\n"
            f"**Parent Comment/Post (that '{main_target_reply_author_name}' replied to in THIS actual task post, ID: '{main_parent_id_ref}')**:\n"
            f"  - Author: '{main_parent_author}'\n"
            f"  - Score: {main_parent_score}\n"  # Provide parent's score
            f"  - Timestamp: {main_parent_timestamp}\n"  # Provide parent's timestamp
            f"  - Body: \"{main_parent_body_full}\"\n\n"  # Provide parent's full body
            f"**Author's Reply to Predict (ID: '{main_target_reply_id_ref}')**:\n"
            f"  - Timestamp: '{main_target_reply_timestamp}'\n"
            f"  - Body: \"{main_target_reply_body_full}\"\n\n"  # Provide reply's full body
            f"{prediction_request_text}\n"  # This contains the "Predict the specific numerical score..." line
            "RULES FOR YOUR RESPONSE (reiteration):\n"
            "- Your goal is to predict the integer score for the Author's Reply in the ACTUAL TASK section.\n"
            "- The score's absolute value MUST be 3 or greater.\n"
            "- Output ONLY the integer. NOTHING ELSE. For example: -7 or 5 or 25.\n\n"
            "Score:"
        )
        output_data_lines_with_convo.append({
            "post_data_store": post_data_store_with_convo,
            # Contains target post (score hidden) AND FS context post (its example score hidden)
            "prediction_task": {
                "prompt_text": prompt_text_for_with_convo_task,
                "target_post_reference_id": target_post_ref_id,  # The post where the prediction happens
                "target_reply_id": main_target_reply_id_ref,  # The reply whose score is predicted
                "true_label": true_numerical_score_for_task,
                "subreddit": current_sample_data["current_subreddit"],
                "few_shot_info": {  # Info about the FS example (which used the context_provider_sample)
                    "type": "pseudo_random_context_distant_post",
                    "fs_context_provider_post_ref_id": fs_details["fs_post_ref_id"],
                    "fs_context_provider_reply_id_hidden_in_example": fs_details["fs_reply_id_hidden"]
                },
                # Adding information about the target reply and its parent for direct access if needed later by user
                "target_reply_body": main_target_reply_body_full,
                "target_reply_parent_info": {
                    "parent_id": main_parent_id_ref,
                    "parent_author": main_parent_author,
                    "parent_score": main_parent_score,
                    "parent_timestamp": main_parent_timestamp,
                    "parent_body": main_parent_body_full
                }
            }
        })
        processed_post_count += 1

    with open(OUTPUT_JSONL_FILE, 'w', encoding='utf-8') as f_out:
        for item in output_data_lines_with_convo:
            f_out.write(json.dumps(item) + '\n')

    print(f"\nProcessing complete.")
    print(f"Generated {len(output_data_lines_with_convo)} entries for '{OUTPUT_JSONL_FILE}'.")
    print(
        f"Total samples initially considered for tasks/context (before +/- filtering): {len(all_valid_samples_for_tasks_and_context)}")
    print(f"Negative samples used: {len(neg_samples)}")
    print(f"Positive samples initially: {len(pos_samples)}, kept after filtering: {len(kept_pos_samples)}")
    print(f"Skipped {skipped_post_count} posts due to initial filtering criteria (before collection for +/- split).")
    print(
        f"Additionally, {deleted_half_positive_low_score_count} positive low-score samples were filtered out from the collected valid samples.")
    print(f"Final processed entries (tasks generated): {processed_post_count}.")


if __name__ == '__main__':
    if not os.path.exists(INPUT_JSON_FILE):
        print(f"CRITICAL ERROR: Input file '{INPUT_JSON_FILE}' not found. Please create it or check the path.")
        sys.exit(1)
    main()