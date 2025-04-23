import pandas as pd
from tabulate import tabulate


def get_latex_table(df):
    table = tabulate(
        df, headers='keys',
        tablefmt='latex', numalign='left', stralign='left',
        showindex=True,
    )
    table = table.replace("\\_", "-")
    return table


if __name__ == "__main__":
    #================================================================
    # Label count
    #================================================================
    l_df = []

    genres = ['shonen', 'shojo', 'seinen', 'josei']

    indices = [
        'Indoors',
        'Outdoors',
        'Day',
        'Night',
        'Sunny',
        'Rainy', 'Snowy',
    ]

    df = pd.read_csv("tasks/recognition_background.csv")
    unique_combinations = df[["circular_index", "task", "label", "genre"]].drop_duplicates()
    unique_combinations = unique_combinations.sort_values(by="circular_index")

    counts = unique_combinations.value_counts(["label", "genre"]).reset_index(name='count')
    pivot_table = counts.pivot(index='label', columns='genre', values='count').fillna(0)
    d = pivot_table[genres].loc[indices]
    l_df.append(d)


    df = pd.read_csv("tasks/character_count.csv")

    counts = df.value_counts(["label", "genre"]).reset_index(name='count')
    pivot_table = counts.pivot(index='label', columns='genre', values='count').fillna(0)
    d = pivot_table[genres]
    l_df.append(d)


    df_stats = pd.concat(l_df)
    df_stats["sum"] = df_stats.sum(axis=1)
    d = df_stats.astype(int)
    d.loc["sum"] = d.sum()
    print(get_latex_table(d))


    # External datasets
    df_genre = pd.read_csv("external/manga_benchmark_dataset/annotations/genre.csv")
    df = pd.read_csv("build/onomatopoeia_scene_panels.csv")
    df = df[df.category=="raw"]
    d = pd.merge(left=df, right=df_genre, on="title", how="inner")
    d.value_counts("genre")

    counts = d.value_counts("genre")
    d = pd.DataFrame(counts).T
    d.index=["Onomatopoeia"]
    l_df.append(d)

    df = pd.read_csv("tasks/emotion_benchmark_split.csv")
    df = df[df.category=="face"]
    unique_combinations = df[["circular_index", "task", "label", "genre"]].drop_duplicates()
    unique_combinations = unique_combinations.sort_values(by="circular_index")

    counts = unique_combinations.value_counts(["label", "genre"]).reset_index(name='count')
    pivot_table = counts.pivot(index='label', columns='genre', values='count').fillna(0)
    d = pivot_table[genres]
    l_df.append(d)

    df_stats = pd.concat(l_df)
    df_stats["sum"] = df_stats.sum(axis=1)
    d = df_stats.astype(int)
    d.loc["sum"] = d.sum()
    print(get_latex_table(d))


    #================================================================
    # Prompt count
    #================================================================
    l_q_counts = []

    df = pd.read_csv("tasks/recognition_background.csv")
    unique_combinations = df[["circular_index","task"]].drop_duplicates()
    unique_combinations = unique_combinations.sort_values(by="circular_index")
    count_circ = unique_combinations.value_counts(["task"]).reset_index(name='count_circ')
    count_circ = count_circ.set_index("task")

    unique_combinations = df[["prompt_id","task"]].drop_duplicates()
    unique_combinations = unique_combinations.sort_values(by="prompt_id")
    count_prompt = unique_combinations.value_counts(["task"]).reset_index(name='count_prompt')
    count_prompt = count_prompt.set_index("task")

    d = pd.concat([count_circ, count_prompt], axis=1)
    l_q_counts.append(d)


    df = pd.read_csv("tasks/character_count.csv")
    unique_combinations = df[["prompt_id","task"]].drop_duplicates()
    unique_combinations = unique_combinations.sort_values(by="prompt_id")
    count_prompt = unique_combinations.value_counts(["task"]).reset_index(name='count_prompt')
    count_prompt = count_prompt.set_index("task")

    unique_combinations = df[["prompt_id","task"]].drop_duplicates()
    unique_combinations = unique_combinations.sort_values(by="prompt_id")
    count_circ = unique_combinations.value_counts(["task"]).reset_index(name='count_circ')
    count_circ = count_circ.set_index("task")

    d = pd.concat([count_circ, count_prompt], axis=1)
    l_q_counts.append(d)


    df = pd.read_csv("tasks/onomatopoeia_scene.csv")
    unique_combinations = df[["circular_id", "category", "is_transcription_shown"]].drop_duplicates()
    unique_combinations = unique_combinations.sort_values(by="circular_id").reset_index(drop=True)
    count_circ = unique_combinations.value_counts(["category", "is_transcription_shown"]).reset_index(name='count_circ')
    count_circ = count_circ.set_index(["category","is_transcription_shown"])

    unique_combinations = df[["prompt_id","category","is_transcription_shown"]].drop_duplicates()
    unique_combinations = unique_combinations.sort_values(by="prompt_id")
    count_prompt = unique_combinations.value_counts(["category","is_transcription_shown"]).reset_index(name='count_prompt')
    count_prompt = count_prompt.set_index(["category","is_transcription_shown"])

    d = pd.concat([count_circ, count_prompt], axis=1)
    l_q_counts.append(d)


    df = pd.read_csv("tasks/emotion_benchmark_split.csv")
    unique_combinations = df[["circular_index", "category"]].drop_duplicates()
    unique_combinations = unique_combinations.sort_values(by="circular_index").reset_index(drop=True)
    count_circ = unique_combinations.value_counts(["category"]).reset_index(name='count_circ')
    count_circ = count_circ.set_index(["category"])

    unique_combinations = df[["prompt_id","category"]].drop_duplicates()
    unique_combinations = unique_combinations.sort_values(by="prompt_id")
    count_prompt = unique_combinations.value_counts(["category"]).reset_index(name='count_prompt')
    count_prompt = count_prompt.set_index(["category"])

    d = pd.concat([count_circ, count_prompt], axis=1)
    l_q_counts.append(d)


    df = pd.read_csv("tasks/panel_localization.csv")
    unique_combinations = df[["circular_id", "category"]].drop_duplicates()
    unique_combinations = unique_combinations.sort_values(by="circular_id").reset_index(drop=True)
    count_circ = unique_combinations.value_counts(["category"]).reset_index(name='count_circ')
    count_circ = count_circ.set_index(["category"])

    unique_combinations = df[["prompt_id","category"]].drop_duplicates()
    unique_combinations = unique_combinations.sort_values(by="prompt_id")
    count_prompt = unique_combinations.value_counts(["category"]).reset_index(name='count_prompt')
    count_prompt = count_prompt.set_index(["category"])

    d = pd.concat([count_circ, count_prompt], axis=1)
    l_q_counts.append(d)


    l_columns = ["category", 'cropped_type','is_with_transcription','image_order_type']
    df = pd.read_csv("tasks/next_panel_inference/test.csv")
    unique_combinations = df[["circular_id"] + l_columns].drop_duplicates()
    unique_combinations = unique_combinations.sort_values(by="circular_id").reset_index(drop=True)
    count_circ = unique_combinations.value_counts(l_columns).reset_index(name='count_circ')
    count_circ = count_circ.set_index(l_columns)

    unique_combinations = df[["prompt_id"]+l_columns].drop_duplicates()
    unique_combinations = unique_combinations.sort_values(by="prompt_id")
    count_prompt = unique_combinations.value_counts(l_columns).reset_index(name='count_prompt')
    count_prompt = count_prompt.set_index(l_columns)

    d = pd.concat([count_circ, count_prompt], axis=1)
    l_q_counts.append(d)


    df_q_counts = pd.concat(l_q_counts)

    print(get_latex_table(df_q_counts))
    print(df_q_counts.sum())
