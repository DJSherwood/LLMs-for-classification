import polars as pl
file_name = 'wlc.txt'

query = (
    pl.scan_csv(
        file_name,
        separator='\t',
        has_header=False,
        new_columns=["Location", "Token", "HebrewWord"],
        schema_overrides={"Location": pl.Utf8, "Token": pl.Utf8, "HebrewWord": pl.Utf8}
    )
    .with_columns(
        pl.col("Location").str.split_exact(" ",1)
        .struct.rename_fields(["Book","ChapterVerseWord"])
        .alias("fields")
    ).unnest("fields")
    .with_columns(
        pl.col("ChapterVerseWord").str.split_exact(":",1)
        .struct.rename_fields(["Chapter","verseWord"])
        .alias("fields")
    ).unnest("fields")
    .with_columns(
        pl.col("verseWord").str.split_exact(".",1)
        .struct.rename_fields(["Verse", "Word"])
        .alias("fields")
    ).unnest("fields")
    .filter(
        ~pl.col("Word").str.contains("[k]")
    ).with_columns(
        pl.col("Word").str.replace_all("[a-zA-Z]","")
    )
    .with_columns(
        pl.when(
            (pl.col("Token") == "?") | (pl.col("Token") == "") | (pl.col("Token").is_null()) | (pl.col("Token") == "null")
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
    ).select(["Token","HebrewWord","Book","Chapter","Verse","Word"])
)

token_list = query.group_by(
    ["Book","Chapter", "Verse"], maintain_order=True
).agg(["Token"])
# print(token_list.collect())

hebrew_word_list = query.group_by(
    ["Book", "Chapter", "Verse"], maintain_order=True
).agg(["HebrewWord"])
# print(hebrew_word_list.collect())

final_df = token_list.join(hebrew_word_list, on=["Book", "Chapter", "Verse"]).sort(by=["Book","Chapter","Verse"]).collect()

display_df = final_df.filter(
    (pl.col("Book") == "Gen") & (pl.col("Chapter") == 1) & (pl.col("Verse") == 1)
)
# print(display_df)
print(f"The Tokens: {display_df[0, "Token"]} Represent the words: {display_df[0, "HebrewWord"]}")

# to do
# need to add the label for which editor is associated with each passage ( for fine-tuning )