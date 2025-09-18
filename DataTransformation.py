import polars as pl
file_name = 'wlc.txt'

q = (
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
    # .with_columns(
    #     pl.col("Location").str.strip_chars(),
    #     pl.col("Token").str.strip_chars(),
    #     pl.col("HebrewWord").str.strip_chars(),
    #     pl.col("Chapter").str.strip_chars().cast(pl.Int8),
    #     pl.col("verseWord").str.strip_chars().cast(pl.Decimal)
    # )
    .filter(pl.col("verseWord").str.contains("[a-zA-Z]"))
)
print(q.collect())

# df = pl.read_csv(file_name, sep='\t', has_header=False)
