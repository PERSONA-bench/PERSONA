#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from collections import defaultdict

def group_posts_by_author(post_list):
    """
    Group posts by their author.

    Args:
        post_list (list[dict]): with key 'author'。

    Returns:
        dict[str, list[dict]]: {author: [post, ...], ...}
    """
    grouped = defaultdict(list)
    for post in post_list:
        author = post.get("author", "UNKNOWN_AUTHOR")
        grouped[author].append(post)
    return grouped

input_filename  = "Raw_Data_Postized.json"
output_filename = "merged_posts_remapped.json"
try:
    with open(input_filename, "r", encoding="utf-8") as f_in:
        flat_posts = json.load(f_in)
    restored_dataset = group_posts_by_author(flat_posts)
    with open(output_filename, "w", encoding="utf-8") as f_out:
        json.dump(restored_dataset, f_out, indent=2, ensure_ascii=False)

    print(f"✔ It's done! Processed file is saved: {output_filename}")
except FileNotFoundError:
    print(f"❌ Input file not found '{input_filename}'。")
except json.JSONDecodeError:
    print(f"❌ Error: '{input_filename}' is not a good json file")
except Exception as e:
    print(f"❌ Something I don't know happended：{e}")
