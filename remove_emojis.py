#!/usr/bin/env python3
"""
Remove all emojis from Reddit parser files
"""

import os
import re

def remove_emojis_from_file(file_path):
    """Remove emojis from a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove all emojis using Unicode ranges
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002600-\U000026FF"  # miscellaneous symbols
            "\U00002700-\U000027BF"  # dingbats
            "]+", flags=re.UNICODE)
        
        original_content = content
        content = emoji_pattern.sub('', content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {file_path}")
            return True
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Remove emojis from all Reddit parser files"""
    parser_dir = "parsers/redditParser"
    
    if not os.path.exists(parser_dir):
        print(f"Directory {parser_dir} not found")
        return
    
    files_processed = 0
    files_fixed = 0
    
    for filename in os.listdir(parser_dir):
        if filename.endswith('.py'):
            file_path = os.path.join(parser_dir, filename)
            files_processed += 1
            if remove_emojis_from_file(file_path):
                files_fixed += 1
    
    print(f"Processed {files_processed} files")
    print(f"Fixed {files_fixed} files")

if __name__ == "__main__":
    main()
