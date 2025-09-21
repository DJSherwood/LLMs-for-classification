import polars as pl
from typing import List, Tuple

file_name = 'wlc.txt'

# static
def parse_range(range_str: str) -> List[int]:
    """Parse a range like '1-3' or a single number '5' into a list of integers."""
    if '-' in range_str:
        start, end = map(int, range_str.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(range_str)]

# static
def parse_passage(passage: str) -> List[Tuple[str, int, List[int]]]:
    """
    Parses a passage like 'Gen 1:1-31' or 'Lev 1–4' into a list of (book, chapter, [verses]) tuples.
    """
    results = []
    if ':' in passage:
        # Format: Book Chapter:Verse(s)
        book_chap, verses = passage.split(':')
        parts = book_chap.split()
        book = ' '.join(parts[:-1])
        chapter = int(parts[-1])
        verse_ranges = [v.strip() for v in verses.split(',')]
        all_verses = []
        for v in verse_ranges:
            if '-' in v:
                all_verses.extend(parse_range(v))
            else:
                all_verses.append(int(v))
        results.append((book, chapter, all_verses))
    else:
        # Format: Book Chapter–Chapter or single Chapter (e.g., 'Lev 1–4' or 'Gen 17')
        parts = passage.split()
        book = ' '.join(parts[:-1])
        chapter_part = parts[-1].replace('–', '-').replace('—', '-')
        if '-' in chapter_part:
            start_ch, end_ch = map(int, chapter_part.split('-'))
            for ch in range(start_ch, end_ch + 1):
                results.append((book, ch, list(range(1, 200))))  # Assumes up to 200 verses
        else:
            ch = int(chapter_part)
            results.append((book, ch, list(range(1, 200))))
    return results

class DataTransformation:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.final_df = None

    def initial_transform(self):
        query = (
            pl.scan_csv(
                self.file_name,
                separator='\t',
                has_header=False,
                new_columns=["Location", "Token", "HebrewWord"],
                schema_overrides={"Location": pl.Utf8, "Token": pl.Utf8, "HebrewWord": pl.Utf8}
            )
            .with_columns(
                pl.col("Location").str.split_exact(" ", 1)
                .struct.rename_fields(["Book", "ChapterVerseWord"])
                .alias("fields")
            ).unnest("fields")
            .with_columns(
                pl.col("ChapterVerseWord").str.split_exact(":", 1)
                .struct.rename_fields(["Chapter", "verseWord"])
                .alias("fields")
            ).unnest("fields")
            .with_columns(
                pl.col("verseWord").str.split_exact(".", 1)
                .struct.rename_fields(["Verse", "Word"])
                .alias("fields")
            ).unnest("fields")
            .filter(
                ~pl.col("Word").str.contains("[k]")
            ).with_columns(
                pl.col("Word").str.replace_all("[a-zA-Z]", "")
            )
            .with_columns(
                pl.when(
                    (pl.col("Token") == "?") | (pl.col("Token") == "") | (pl.col("Token").is_null()) | (
                                pl.col("Token") == "null")
                ).then(
                    pl.lit("9999")
                ).otherwise(
                    pl.col("Token")
                ).alias("Token")
            )
            .with_columns(
                pl.col("Token").str.strip_chars().cast(pl.Int32),
                pl.col("HebrewWord").str.strip_chars(),
                pl.col("Book").str.strip_chars(),
                pl.col("Chapter").str.strip_chars().cast(pl.Int32),
                pl.col("Verse").str.strip_chars().cast(pl.Int32),
                pl.col("Word").str.strip_chars().cast(pl.Int32)
            ).sort(
                ["Book", "Chapter", "Verse", "Word"], descending=[False, False, False, True]
            ).select(["Token", "HebrewWord", "Book", "Chapter", "Verse", "Word"])
        )

        token_list = query.group_by(
            ["Book", "Chapter", "Verse"], maintain_order=True
        ).agg(["Token"])
        # print(token_list.collect())

        hebrew_word_list = query.group_by(
            ["Book", "Chapter", "Verse"], maintain_order=True
        ).agg(["HebrewWord"])
        # print(hebrew_word_list.collect())

        self.final_df = token_list.join(hebrew_word_list, on=["Book", "Chapter", "Verse"]).sort(
            by=["Book", "Chapter", "Verse"]).collect()

        return self.final_df

    def parse_all_passages(passages: List[str]) -> List[Tuple[str, int, int]]:
        """
        Converts a list of passage strings into (book, chapter, verse) tuples.
        """
        all_refs = []
        for p in passages:
            parsed = parse_passage(p)
            for book, chapter, verses in parsed:
                for v in verses:
                    all_refs.append((book, chapter, v))
        return all_refs







passages = [
    "Gen 1:1-31", "Gen 2:1-3", "Gen 5:3-28", "Gen 5:30-32",
    "Gen 6:9-22", "Gen 9:1-17", "Gen 9:28-29", "Gen 10:2-7",
    "Gen 10:20", "Gen 10:22-23", "Gen 10:31",
    "Lev 1–4", "Lev 8–9", "Exod 25–31", "Exod 35–40", "Gen 17"
]

refs = parse_all_passages(passages)

# Preview first few
for ref in refs[:10]:
    print(ref)

import polars as pl

priest_refs = set(parse_all_passages(priest_passages))  # define priest_passages list first

df = df.with_columns([
    pl.when(pl.struct(["book", "chapter", "verse"]).map_elements(lambda row: (row["book"], row["chapter"], row["verse"]) in deut_refs))
      .then("Deut")
      .when(pl.struct(["book", "chapter", "verse"]).map_elements(lambda row: (row["book"], row["chapter"], row["verse"]) in deuthist_refs))
      .then("DeutHist")
      .when(pl.struct(["book", "chapter", "verse"]).map_elements(lambda row: (row["book"], row["chapter"], row["verse"]) in priest_refs))
      .then("Priest")
      .otherwise(None)
      .alias("editor")
])
