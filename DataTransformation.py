import polars as pl
import re
import torch

# static
def parse_passages(passage_dict):
    from collections import defaultdict

    editor_map = defaultdict(set)

    for editor, refs in passage_dict.items():
        for ref in refs:
            # Match Book name and the rest
            match = re.match(r"^(\w+)\s+(.+)$", ref)
            if not match:
                continue
            book, rest = match.groups()

            if ':' in rest:
                # Format like 1:1-31 or 14:1-4
                chap_verse = rest.split(":")
                chapter = int(chap_verse[0])
                verses = chap_verse[1]

                if '-' in verses:
                    start_verse, end_verse = map(int, verses.split("-"))
                    for verse in range(start_verse, end_verse + 1):
                        editor_map[editor].add((book, chapter, verse))
                else:
                    editor_map[editor].add((book, chapter, int(verses)))

            elif '-' in rest:
                # Chapter range like "Deut 12-13"
                start_chap, end_chap = map(int, rest.split("-"))
                for chapter in range(start_chap, end_chap + 1):
                    editor_map[editor].add((book, chapter, None))
            else:
                # Single full chapter, like "Deut 6" or "Gen 17"
                editor_map[editor].add((book, int(rest), None))

    return editor_map

class DataTransformation:
    def __init__(self, file_name='wlc.txt', passages=None):
        self.file_name = file_name
        self.passage_sets = parse_passages(passages)
        self.final_df = None
        self.df = None

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

    def get_editor(self, book, chapter, verse):
        for editor, refs in self.passage_sets.items():
            if (book, chapter, verse) in refs or (book, chapter, None) in refs:
                return editor
        return None

    def assign_editors(self):
        self.df = self.final_df.with_columns([
            pl.struct(["Book", "Chapter", "Verse"]).map_elements(
                lambda row: self.get_editor(row["Book"], row["Chapter"], row["Verse"]),
                return_dtype=pl.Utf8
            ).alias("editor")
        ])

        return self.df

    def convert_to_torch(self, column_name='Token'):
        flat_list = [item for sublist in self.df[column_name].to_list() for item in sublist]
        flat_tensor = torch.tensor(flat_list, dtype=torch.long)

        return flat_tensor

if __name__ == "__main__":
    # define passages to transform
    passages = {
        "D": ["Deut 6", "Deut 12-13", "Deut 15-16", "Deut 18-19", "Deut 26", "Deut 28"],
        "DH": ["Deut 8-11", "Deut 27", "Josh 1", "Josh 5", "Josh 6", "Josh 12", "Josh 23",
               "Judg 2", "Judg 6", "2Sam 7", "1Kgs 8", "2Kgs 17:1-21", "2Kgs 22-25"],
        "P": ["Gen 1:1-31", "Gen 2:1-3", "Gen 5:3-28","Gen 5:30-32", "Gen 6:9-22", "Gen 9:1-17",
              "Gen 6:28-29", "Gen 10:2-7","Gen 10:20","Gen 10:22-23","Gen 10:31", "Gen 11:11-26",
              "Gen 11:29-32", "Gen 12:5", "Gen 13:6","Gen 13:12", "Gen 16:3","Gen 16:15-16", "Gen 21:2-5",
              "Gen 22:20-24", "Gen 23:1-20", "Gen 25:7-10","Gen 25:13-17","Gen 25:20", "Gen 26:20",
              "Gen 26:34-35", "Gen 27:46", "Gen 28:1-9", "Gen 35:9-15","Gen 35:27-29", "Gen 36:40-43",
              "Gen 37:1", "Gen 46:6-7", "Gen 47:28", "Gen 49:29-33", "Gen 50:12-13", "Exod 1:1-4","Exod 1:7",
              "Exod 1:13-14", "Exod 2:23-25", "Exod 7:1-13","Exod 7:19-22", "Exod 8:1-3","Exod 8:11-15",
              "Exod 9:8-12", "Exod 11:9-10", "Exod 12:40-42","Exod 13:20", "Exod 14:1-4","Exod 14:8-10",
              "Exod 14:15-18","Exod 14:21-23","Exod 14:27-29", "Exod 15:22", "Exod 19:1", "Exod 24:16-17",
              "Gen 17", "Exod 6","Exod 16", "Exod 25-31", "Exod 35-40", "Lev 1-4","Exod 8-9"]
    }

    # Step through the procedure
    dt = DataTransformation(file_name='wlc.txt',passages=passages)
    dt.initial_transform()
    print(dt.final_df.select(pl.col("Token").n_unique()))
    #dt.assign_editors()
    #ft = dt.convert_to_torch(column_name='Token')
