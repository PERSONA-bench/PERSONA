import json
import os
import sys
from datetime import datetime, timezone
from copy import deepcopy
from collections import deque
import random

# --- Configuration ---
INPUT_JSON_FILE = "merged_posts_remapped.json"
OUTPUT_JSONL_FILE = "WithConversationPrompts_ScorePrediction_PseudoRandomContext.jsonl"

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
            elif k.startswith('_') and k not in ['__sub__', '_parsed_timestamp', '_abs_score', '_parent_id',
                                                 '_parent_body', '_parent_author', '_parent_score',
                                                 '_parent_timestamp']:  # Preserve specific internal keys if needed
                continue
            else:
                new_dict[k] = convert_datetimes_to_strings_in_obj(v)
        return new_dict
    return obj


def generate_unique_ids_for_post(post_data, post_index_prefix):
    # This function remains largely the same as it's crucial for identifying nodes.
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
                comment['parent_id'] = post_id_str  # Ensure top-level comments have parent_id pointing to post
                q.append((comment, post_id_str))  # Store with parent_id for clarity

    processed_comments_for_flat_list = []

    def _assign_ids_recursive(comments_list, parent_id_str_rec):
        nonlocal comment_counter
        if not isinstance(comments_list, list):
            return
        for i, comment in enumerate(comments_list):
            if not isinstance(comment, dict):
                continue

            current_id_str = f"{parent_id_str_rec}-c{comment_counter}"  # Simple increment for now
            comment_counter += 1

            comment['id'] = current_id_str
            comment['parent_id'] = parent_id_str_rec  # Set parent_id
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
                parent_body = parent_node.get('content', "")  # Post content is the "parent body"
                parent_author = parent_node.get('author', "")
                parent_score = parent_node.get('score')
                parent_timestamp = parent_node.get('timestamp')
            elif parent_node and parent_node.get('type') == 'comment':
                parent_body = parent_node.get('body', "")
                parent_author = parent_node.get('author', "")
                parent_score = parent_node.get('score')
                parent_timestamp = parent_node.get('timestamp')
            else:
                # print(f"Warning: Parent node not found or invalid type for comment {node.get('id')}", file=sys.stderr)
                continue  # Skip if parent is not found or not a post/comment

            if not parent_body or clean_text(parent_body).lower() in ['[deleted]', '[removed]', '']:
                continue

            parsed_ts = parse_timestamp(node.get('timestamp'))
            if parsed_ts:  # Ensure timestamp is valid
                reply_details = deepcopy(node)  # Keep all original fields from the reply node
                reply_details['_parsed_timestamp'] = parsed_ts
                reply_details['_abs_score'] = abs(score)
                reply_details['_parent_id'] = parent_id
                reply_details['_parent_body'] = clean_text(parent_body)
                reply_details['_parent_author'] = clean_text(parent_author)
                reply_details['_parent_score'] = parent_score  # Store parent's score
                reply_details['_parent_timestamp'] = parent_timestamp  # Store parent's timestamp
                author_replies.append(reply_details)

    author_replies.sort(key=lambda r: (r['_abs_score'], r['_parsed_timestamp']), reverse=True)
    return author_replies


def get_node_by_id(post_data_struct, node_id):
    """Helper to find a specific node (post or comment) by its ID in a structured post."""
    if post_data_struct.get('id') == node_id:
        return post_data_struct

    q = deque()
    if 'comments' in post_data_struct and isinstance(post_data_struct.get('comments'), list):
        for comment in post_data_struct['comments']:
            if isinstance(comment, dict): q.append(comment)

    while q:
        node = q.popleft()
        if not isinstance(node, dict): continue
        if node.get('id') == node_id:
            return node
        if 'replies' in node and isinstance(node['replies'], list):
            for reply in node['replies']:
                if isinstance(reply, dict): q.append(reply)
    return None


def build_minimal_context_for_prediction(full_post_data_with_ids, target_reply_node_obj, parent_node_obj):
    """
    Creates a structure for post_data_store containing only the target reply (score hidden)
    and its direct parent.
    """
    # Deepcopy to avoid modifying original objects, especially the target_reply_node_obj
    target_reply_copy = deepcopy(target_reply_node_obj)
    parent_copy = deepcopy(parent_node_obj)

    # Crucially, remove the score from the target reply *copy*
    if 'score' in target_reply_copy:
        target_reply_copy['score'] = "[SCORE_TO_PREDICT]"
    if '_abs_score' in target_reply_copy:  # Also remove internal score representations
        del target_reply_copy['_abs_score']
    if '_parsed_timestamp' in target_reply_copy:  # Clean up internal fields not needed for LLM
        del target_reply_copy['_parsed_timestamp']

    # Ensure parent_copy and target_reply_copy have their essential fields
    # and remove replies/comments from them to keep the context minimal.
    if 'comments' in parent_copy:  # If parent is a post
        parent_copy['comments'] = []
    if 'replies' in parent_copy:  # If parent is a comment
        parent_copy['replies'] = []
    if 'replies' in target_reply_copy:
        target_reply_copy['replies'] = []

    # Create the structure for post_data_store
    # The 'post' will act as a container. Its ID will be the original post's ID.
    # The actual content of this container post isn't the full original post,
    # but rather a way to structure the parent-reply relationship.

    context_store_entry = {
        "id": full_post_data_with_ids['id'],  # Original post ID
        "type": "post",
        "author": full_post_data_with_ids.get('author'),
        "title": full_post_data_with_ids.get('title'),
        "__sub__": full_post_data_with_ids.get('__sub__'),
        "comments": []  # Initialize comments
    }

    if parent_copy['type'] == 'post':
        # This case is tricky. If the target reply is a direct reply to the post,
        # then the parent_copy IS the post.
        # We want to store the post (parent_copy) and then the target_reply_copy as its comment.
        context_store_entry = parent_copy  # The parent is the post itself
        context_store_entry['comments'] = [target_reply_copy]
        # Ensure the post's score is present (not hidden, unless it's the target itself, which is not this function's primary design)
        if 'score' not in context_store_entry and 'score' in full_post_data_with_ids:
            context_store_entry['score'] = full_post_data_with_ids['score']


    elif parent_copy['type'] == 'comment':
        # Parent is a comment. Target reply is a reply to this comment.
        # The context_store_entry (post) will contain the parent_comment,
        # and the parent_comment will contain the target_reply.
        parent_copy['replies'] = [target_reply_copy]
        context_store_entry['comments'] = [parent_copy]
        # Add some minimal post details to the context_store_entry
        context_store_entry['score'] = full_post_data_with_ids.get('score')  # Score of the original post
        context_store_entry['timestamp'] = full_post_data_with_ids.get('timestamp')
        context_store_entry['content'] = "Contextual post information. The key interaction is within the comments."

    # Ensure all datetime objects are strings
    return convert_datetimes_to_strings_in_obj(context_store_entry)


def build_dialogue_tree_with_hidden_score_for_few_shot(post_data_with_ids, node_id_to_hide_score_of):
    """
    Prepares a full post dialogue tree for few-shot examples, hiding the score of a specific node.
    This is similar to the original build_dialogue_tree_with_hidden_score.
    """
    context_data = deepcopy(post_data_with_ids)
    q = deque()

    if context_data.get('id') == node_id_to_hide_score_of and 'score' in context_data:
        context_data['score'] = "[SCORE_TO_PREDICT]"

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
    return convert_datetimes_to_strings_in_obj(context_data)


def get_post_summary_string_for_prompt(post_obj, is_few_shot_unrelated=False):
    title = clean_text(post_obj.get('title', "N/A"))
    subreddit = clean_text(post_obj.get('__sub__', "N/A"))
    author = clean_text(post_obj.get('author', "N/A"))
    # Score of the post itself, not a reply
    score = post_obj.get('score', "N/A") if post_obj.get('type') == 'post' else "N/A (comment)"
    timestamp = datetime_to_string(parse_timestamp(post_obj.get('timestamp'))) or "N/A"
    content_full = clean_text(post_obj.get('content', post_obj.get('body', "N/A")))  # Handles posts and comments
    content_summary = content_full[:150] + ('...' if len(content_full) > 150 else '')  # Shorter summary for prompt
    post_id = post_obj.get('id', "N/A")

    header = f"### FEW-SHOT EXAMPLE (Context from a DIFFERENT, unrelated post ID: {post_id}) ###" if is_few_shot_unrelated \
        else f"### CURRENT TASK (Post ID: {post_id}) ###"

    return (
        f"{header}\n"
        f"- Subreddit: r/{subreddit}\n"
        f"- Post Author: u/{author}\n"  # Assuming this is called for the main post context
        f"- Post Score: {score}\n"
        f"- Post Timestamp: {timestamp}\n"
        f"- Post Title: \"{title}\"\n"
        f"- Post Content Summary: \"{content_summary}\"\n"
        # f"(Full context for this item is in the data store under ID '{post_id}')" # Removed to save space in prompt
    )


# --- Main Processing ---
def main():
    global skipped_post_count, processed_post_count, deleted_half_positive_low_score_count

    if not os.path.exists(INPUT_JSON_FILE):
        print(f"Error: Input file '{INPUT_JSON_FILE}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
        all_user_data = json.load(f)

    output_data_lines_with_pseudo_random_conv = []

    # Store all posts with their original reference IDs and necessary structured data
    # This will be used to pick few-shot examples from later posts
    all_processed_posts_for_few_shot_pool = []
    temp_post_idx_counter = 0  # Global index for all posts across all users

    print("Phase 1: Initial processing and preparing pool for few-shot examples...")
    for main_author_name, posts_by_author_orig in all_user_data.items():
        for current_post_data_original_ref in posts_by_author_orig:
            temp_post_idx_counter += 1
            # Use a temporary, globally unique ID for initial processing before final reference ID.
            # This post_reference_id is for internal tracking during this pre-processing.
            post_reference_id_internal = f"internal_p{temp_post_idx_counter}"

            current_post_structured_with_ids = deepcopy(current_post_data_original_ref)
            # Assign IDs; these IDs are relative to this specific post_reference_id_internal
            flat_nodes_of_current_post = generate_unique_ids_for_post(current_post_structured_with_ids,
                                                                      post_reference_id_internal)
            # Convert datetimes early
            current_post_structured_with_ids = convert_datetimes_to_strings_in_obj(current_post_structured_with_ids)

            author_replies_for_score_pred = find_author_replies_in_post_for_score_prediction(
                flat_nodes_of_current_post, main_author_name
            )

            # Basic filtering (can be adjusted)
            initial_skip = False
            participants = set(node.get('author') for node in flat_nodes_of_current_post if node.get('author'))
            if len(participants) <= 2:  # Reduced from 3 for more data, adjust as needed
                initial_skip = True

            if not initial_skip:
                num_total_author_replies = sum(1 for node in flat_nodes_of_current_post if
                                               node.get('type') == 'comment' and
                                               node.get('author') == main_author_name and
                                               node.get('body') and
                                               clean_text(node.get('body')).lower() not in ['', '[deleted]',
                                                                                            '[removed]'])
                if num_total_author_replies < 1:  # Reduced from 2
                    initial_skip = True

            if not initial_skip and not author_replies_for_score_pred:
                initial_skip = True

            if not initial_skip:
                # We need at least one valid author reply to be a candidate for few-shot or actual task
                highest_score_reply = author_replies_for_score_pred[0]
                if highest_score_reply['_abs_score'] <= 1:  # Stricter filter for potentially "noisy" low-score replies
                    initial_skip = True
                elif not highest_score_reply['_parent_body'] or \
                        clean_text(highest_score_reply['_parent_body']).lower() in ['[deleted]', '[removed]', '']:
                    initial_skip = True

            if initial_skip:
                # We don't increment skipped_post_count here as this is a pre-filter for the pool
                continue

            # Store the necessary data for potential use as a few-shot example
            # The 'current_post_structured_with_ids' is the full dialogue tree for this post
            all_processed_posts_for_few_shot_pool.append({
                "post_data_full_struc": current_post_structured_with_ids,  # Full tree with its internal IDs
                "author_name": main_author_name,
                "potential_replies_for_fs": author_replies_for_score_pred,  # List of author's replies in this post
                "internal_ref_id": post_reference_id_internal  # The ID used when generating its tree
            })

    if not all_processed_posts_for_few_shot_pool:
        print("Error: No posts available in the pool for few-shot examples after initial filtering. Exiting.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Phase 1 complete. {len(all_processed_posts_for_few_shot_pool)} posts available for few-shot pool.")

    # Now, iterate through posts again to generate tasks, using the pool for few-shot examples.
    # We re-iterate all_user_data to maintain the original processing order for the *actual tasks*.

    all_task_entries = []  # Store all valid tasks before filtering positives
    current_post_global_index = -1  # To track index for selecting future few-shot posts

    user_g_idx = 0
    print("Phase 2: Generating tasks with pseudo-random few-shot contexts...")
    for main_author_name, posts_by_author_orig in all_user_data.items():
        user_g_idx += 1
        post_user_idx = 0
        for current_post_data_original_ref in posts_by_author_orig:
            post_user_idx += 1
            current_post_global_index += 1  # Increment for each post processed as a potential *task*

            # This is the unique reference ID for the *current task's post*
            task_post_reference_id = f"task_u{user_g_idx}p{post_user_idx}"

            # Re-process the current post to ensure fresh state and correct IDing for the task
            task_post_structured = deepcopy(current_post_data_original_ref)
            task_flat_nodes = generate_unique_ids_for_post(task_post_structured, task_post_reference_id)
            task_post_structured = convert_datetimes_to_strings_in_obj(task_post_structured)

            task_author_replies = find_author_replies_in_post_for_score_prediction(
                task_flat_nodes, main_author_name
            )

            # --- Filtering for the *actual task* post (similar to original script) ---
            initial_skip = False
            task_participants = set(node.get('author') for node in task_flat_nodes if node.get('author'))
            if len(task_participants) <= 3:  # Original filter value
                initial_skip = True

            if not initial_skip:
                num_total_task_author_replies = sum(1 for node in task_flat_nodes if
                                                    node.get('type') == 'comment' and
                                                    node.get('author') == main_author_name and
                                                    node.get('body') and
                                                    clean_text(node.get('body')).lower() not in ['', '[deleted]',
                                                                                                 '[removed]'])
                if num_total_task_author_replies <= 2:  # Original filter value
                    initial_skip = True

            if not initial_skip and not task_author_replies:
                initial_skip = True

            if not initial_skip:
                # The target reply for the *actual task*
                target_reply_for_task = task_author_replies[0]
                if target_reply_for_task['_abs_score'] <= 3:  # Original filter value
                    initial_skip = True
                elif not target_reply_for_task['_parent_body'] or \
                        clean_text(target_reply_for_task['_parent_body']).lower() in ['[deleted]', '[removed]', '']:
                    initial_skip = True

            if initial_skip:
                skipped_post_count += 1
                continue
            # --- End filtering for the *actual task* post ---

            # Target reply for the current prediction task
            task_target_reply_obj = task_author_replies[0]
            task_true_label_score_sentiment = "positive" if task_target_reply_obj['score'] > 0 else "negative"
            task_target_reply_id = task_target_reply_obj['id']

            # Find the parent of the task_target_reply_obj within task_post_structured
            # The parent_id is already in task_target_reply_obj['_parent_id']
            task_parent_node_obj = get_node_by_id(task_post_structured, task_target_reply_obj['_parent_id'])
            if not task_parent_node_obj:
                # This should ideally not happen if find_author_replies worked correctly
                print(
                    f"Warning: Parent node for task reply {task_target_reply_id} not found in {task_post_reference_id}. Skipping.",
                    file=sys.stderr)
                skipped_post_count += 1
                continue

            # --- Prepare Few-Shot Example from a "Future" Post ---
            few_shot_post_data_store_entry = None
            fs_prompt_segment = "### NO SUITABLE FEW-SHOT EXAMPLE FOUND ###\n"  # Default if none can be made
            fs_true_answer_for_prompt = "positive"  # Default, will be overridden

            # Determine the index for the few-shot post (200 posts after current, or wrapped around)
            fs_pool_size = len(all_processed_posts_for_few_shot_pool)
            if fs_pool_size > 0:  # Ensure pool is not empty
                few_shot_candidate_post_index = (current_post_global_index + 200) % fs_pool_size

                # Get the selected post from the pool for few-shot
                fs_candidate_post_bundle = all_processed_posts_for_few_shot_pool[few_shot_candidate_post_index]

                fs_post_full_struc = fs_candidate_post_bundle["post_data_full_struc"]  # Full tree
                fs_author_name = fs_candidate_post_bundle["author_name"]
                fs_potential_replies = fs_candidate_post_bundle["potential_replies_for_fs"]
                fs_post_internal_ref_id = fs_candidate_post_bundle[
                    "internal_ref_id"]  # This is the ID used for its tree nodes

                # Select a reply from this few-shot post to be the example
                # For simplicity, we'll try to use its highest absolute score reply if valid
                if fs_potential_replies:
                    fs_example_reply_obj = fs_potential_replies[0]  # Highest abs score one
                    fs_example_reply_id = fs_example_reply_obj['id']
                    fs_example_reply_true_label = "positive" if fs_example_reply_obj['score'] > 0 else "negative"

                    # The fs_post_full_struc already has datetime strings due to pre-processing.
                    # We need to hide the score of fs_example_reply_id in this structure for the data store.
                    few_shot_post_data_store_entry = build_dialogue_tree_with_hidden_score_for_few_shot(
                        deepcopy(fs_post_full_struc),  # Deepcopy to avoid modifying the pool
                        fs_example_reply_id
                    )
                    # The ID for this few-shot post in the data store will be its original internal_ref_id
                    fs_post_ref_id_for_datastore = fs_post_internal_ref_id

                    # Create prompt segment for this few-shot example
                    fs_post_summary_str = get_post_summary_string_for_prompt(fs_post_full_struc,
                                                                             is_few_shot_unrelated=True)

                    fs_parent_author = fs_example_reply_obj['_parent_author']
                    fs_parent_id_ref = fs_example_reply_obj['_parent_id']  # ID relative to fs_post_internal_ref_id
                    fs_parent_body_summary = clean_text(fs_example_reply_obj['_parent_body'])[
                                             :100] + "..."  # Parent of fs_example_reply_obj

                    fs_reply_body_summary = clean_text(fs_example_reply_obj['body'])
                    fs_reply_timestamp = datetime_to_string(fs_example_reply_obj['_parsed_timestamp'])

                    fs_prompt_segment = (
                        f"{fs_post_summary_str}\n"
                        f"In that unrelated post, its author '{fs_author_name}' made a reply.\n"
                        f"Parent Comment/Post (that '{fs_author_name}' replied to in that unrelated post, ID: '{fs_parent_id_ref}'):\n"  # Clarify ID context
                        f"  - Author: '{fs_parent_author}'\n"
                        f"  - Body/Content Summary: \"{fs_parent_body_summary}\"\n"
                        f"Author's Reply (in that unrelated post, ID: '{fs_example_reply_id}'):\n"  # Clarify ID context
                        f"  - Timestamp: '{fs_reply_timestamp}'\n"
                        f"  - Body: \"{fs_reply_body_summary}\"\n"
                        f"(The actual score of this example reply '{fs_example_reply_id}' from the unrelated post is hidden in its data store entry, but for this example, we will tell you.)\n\n"
                        f"Based on the information from that UNRELATED post, the predicted score sentiment for '{fs_author_name}'s reply (ID: '{fs_example_reply_id}') was:"
                    )
                    fs_true_answer_for_prompt = fs_example_reply_true_label
                else:  # No suitable replies in the selected few-shot post
                    few_shot_post_data_store_entry = None  # Mark as none
                    fs_post_ref_id_for_datastore = "N/A_FS_NOT_FOUND"

            # --- Prepare `post_data_store` for the CURRENT TASK ---
            # This store will contain the minimal context for the actual prediction.
            task_post_data_store = {}

            # Add the minimal context for the current task's prediction
            # This function call will hide the score of task_target_reply_obj
            minimal_task_context_for_store = build_minimal_context_for_prediction(
                task_post_structured,  # The full structure of the current task's post
                task_target_reply_obj,  # The reply whose score we want to predict
                task_parent_node_obj  # The parent of that reply
            )
            task_post_data_store[task_post_reference_id] = minimal_task_context_for_store

            # Add the few-shot post to the data store IF it was successfully created
            if few_shot_post_data_store_entry and fs_post_ref_id_for_datastore != "N/A_FS_NOT_FOUND":
                # Ensure the ID for the few-shot post in the store doesn't clash with the task post ID
                # (fs_post_ref_id_for_datastore is based on internal_pX, task_post_reference_id is task_uXpX)
                task_post_data_store[fs_post_ref_id_for_datastore] = few_shot_post_data_store_entry

            # --- Create Prompt for the ACTUAL TASK ---
            # Summary of the current task's post (not the few-shot one)
            task_main_post_summary_str = get_post_summary_string_for_prompt(task_post_structured)

            # Parent of the reply we are trying to predict (task_target_reply_obj)
            task_parent_author = task_target_reply_obj['_parent_author']
            task_parent_id = task_target_reply_obj['_parent_id']  # ID relative to task_post_reference_id
            task_parent_body_cleaned = clean_text(task_target_reply_obj['_parent_body'])
            task_parent_score_info = task_target_reply_obj['_parent_score']  # Score of the parent
            task_parent_timestamp_info = datetime_to_string(parse_timestamp(task_target_reply_obj['_parent_timestamp']))

            # The reply we are trying to predict (task_target_reply_obj)
            # Its score is NOT included here or in the data store for this task.
            task_reply_body_cleaned = clean_text(task_target_reply_obj['body'])
            task_reply_timestamp = datetime_to_string(task_target_reply_obj['_parsed_timestamp'])
            # task_target_reply_id is already defined

            # Construct the final prompt
            prompt_text = (
                f"{fs_prompt_segment}\n{fs_true_answer_for_prompt}\n\n"  # Few-shot example and its answer
                "--- IMPORTANT NOTE: The few-shot example above is from a COMPLETELY UNRELATED post. "
                "Its content and context DO NOT directly relate to the actual task below. "
                "We are studying the effect of such pseudo-random contextual prompts. ---\n\n"
                f"{task_main_post_summary_str}\n\n"
                f"Now, for the actual task, consider the post author '{main_author_name}'. This author made a reply within this current post (ID: '{task_post_reference_id}').\n"
                f"The author's reply was made to the following parent comment/post (ID: '{task_parent_id}'):\n"
                f"  - Parent Author: u/{task_parent_author}\n"
                f"  - Parent Score: {task_parent_score_info}\n"
                f"  - Parent Timestamp: {task_parent_timestamp_info}\n"
                f"  - Parent Body/Content: \"{task_parent_body_cleaned}\"\n\n"  # Give full parent body
                f"The Author's Reply (ID: '{task_target_reply_id}', for which you need to predict the score sentiment) is:\n"
                f"  - Reply Timestamp: {task_reply_timestamp}\n"
                f"  - Reply Body: \"{task_reply_body_cleaned}\"\n\n"  # Give full reply body
                f"Given this information about the CURRENT post and the specific reply by '{main_author_name}', "
                f"predict if the score for this author's reply (ID: '{task_target_reply_id}') was positive or negative.\n"
                f"Respond with only 'positive' or 'negative'.\n"
                "Predicted score sentiment:"
            )

            current_task_entry = {
                "post_data_store": task_post_data_store,  # Contains minimal current task context + FS post context
                "prediction_task": {
                    "prompt_text": prompt_text,
                    "target_post_reference_id": task_post_reference_id,
                    # Refers to the post in the store holding the minimal task data
                    "target_reply_id": task_target_reply_id,
                    # The ID of the reply whose score is hidden and needs prediction
                    "true_label": task_true_label_score_sentiment,
                    "subreddit": clean_text(task_post_structured.get('__sub__', "N/A")),
                    "few_shot_info": {
                        "fs_post_ref_id_in_store": fs_post_ref_id_for_datastore if few_shot_post_data_store_entry else "N/A_FS_NOT_FOUND",
                        "fs_example_reply_id_in_fs_post": fs_example_reply_id if few_shot_post_data_store_entry and fs_potential_replies else "N/A"
                    }
                },
                # For filtering positive samples later
                "_abs_score_for_filtering": task_target_reply_obj['_abs_score'],
                "_is_positive_sample": (task_true_label_score_sentiment == "positive")
            }
            all_task_entries.append(current_task_entry)

    # --- Filter positive samples (similar to original script's logic) ---
    print(f"\nPhase 2 complete. {len(all_task_entries)} raw task entries generated.")
    print("Phase 3: Filtering positive samples...")

    positive_samples = [entry for entry in all_task_entries if entry["_is_positive_sample"]]
    negative_samples = [entry for entry in all_task_entries if not entry["_is_positive_sample"]]

    positive_samples.sort(key=lambda x: x["_abs_score_for_filtering"])

    num_positive_to_delete = 0
    if len(positive_samples) > 0:  # Avoid division by zero or negative slice if list is small
        num_positive_to_delete = len(positive_samples) // 2  # Delete half of the lowest score positive samples

    deleted_half_positive_low_score_count = num_positive_to_delete
    skipped_post_count += num_positive_to_delete  # Add to general skipped count

    final_positive_samples_to_keep = positive_samples[num_positive_to_delete:]

    # Combine kept positive samples and all negative samples
    final_output_entries = final_positive_samples_to_keep + negative_samples
    random.shuffle(final_output_entries)  # Shuffle the final list for good measure

    processed_post_count = len(final_output_entries)

    # Clean up temporary keys from final entries before writing
    for entry in final_output_entries:
        del entry["_abs_score_for_filtering"]
        del entry["_is_positive_sample"]

    with open(OUTPUT_JSONL_FILE, 'w', encoding='utf-8') as f_out:
        for item in final_output_entries:
            f_out.write(json.dumps(item) + '\n')

    print(f"\nPhase 3 complete. Processing finished.")
    print(f"Generated {len(final_output_entries)} entries for '{OUTPUT_JSONL_FILE}'.")
    print(f"Total posts processed into final output: {processed_post_count}.")
    print(
        f"Total posts skipped due to various filtering criteria (including {deleted_half_positive_low_score_count} positive low-score samples): {skipped_post_count}.")


if __name__ == '__main__':
    if not os.path.exists(INPUT_JSON_FILE):
        print(f"CRITICAL ERROR: Input file '{INPUT_JSON_FILE}' not found. Please create it or check the path.")
        sys.exit(1)
    main()