import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_gbq
from google.colab import auth


def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = name.replace("*", "")
    name = re.sub(r"(?<=\b[a-z])\.", "", name)
    name = re.sub(r"\s+(jr\.?|sr\.?|iii|ii|iv)$", "", name)
    return " ".join(name.split())


def determine_position(row):
    positions = []
    if row.get("guard") == 1:
        positions.append("Guard")
    if row.get("forward") == 1:
        positions.append("Forward")
    if row.get("center") == 1:
        positions.append("Center")
    if not positions:
        return "Unknown"
    return "-".join(positions)


def load_clean_nba_draft_data():
    uncleaned_draft_data = pd.read_csv("/content/nbaplayersdraft.csv")
    years_played_mask = uncleaned_draft_data["years_active"] >= 1
    average_minutes_mask = uncleaned_draft_data["average_minutes_played"] >= 5
    draft_year_top_mask = uncleaned_draft_data["year"] <= 2017
    draft_year_bottom_mask = uncleaned_draft_data["year"] >= 2013

    cleaned_draft_data = uncleaned_draft_data[
        years_played_mask
        & average_minutes_mask
        & draft_year_top_mask
        & draft_year_bottom_mask
    ].copy()
    cleaned_draft_data["college"] = cleaned_draft_data["college"].fillna(
        "None"
    )
    cleaned_draft_data["player_clean"] = cleaned_draft_data["player"].apply(
        normalize_name
    )

    players_df = pd.read_csv("/content/Players.csv")
    players_df["player"] = (
        players_df["firstName"].fillna("")
        + " "
        + players_df["lastName"].fillna("")
    ).str.strip()
    players_df["player_clean"] = players_df["player"].apply(normalize_name)

    players_df["position"] = players_df.apply(determine_position, axis=1)
    players_pos_df = players_df[["player_clean", "position"]].copy()
    players_pos_df["is_unknown"] = players_pos_df["position"] == "Unknown"
    players_pos_df = players_pos_df.sort_values("is_unknown")
    players_pos_df = players_pos_df.drop_duplicates(
        subset=["player_clean"], keep="first"
    )
    players_pos_df = players_pos_df.drop(columns=["is_unknown"])

    cleaned_draft_data = pd.merge(
        cleaned_draft_data, players_pos_df, on="player_clean", how="left"
    )
    cleaned_draft_data["position"] = cleaned_draft_data["position"].fillna(
        "Unknown"
    )
    cleaned_draft_data = cleaned_draft_data.drop(columns=["player_clean"])
    return cleaned_draft_data


def fetch_college_statistics(player_names, project_id):
    query_advanced = f"""
    SELECT
      full_name AS player,
      COUNT(DISTINCT game_id) AS games_played,
      SUM(points) AS pts,
      SUM(rebounds) AS reb,
      SUM(assists) AS ast,
      SUM(steals) AS stl,
      SUM(blocks) AS blk,
      SUM(turnovers) AS tov,
      SUM(field_goals_att) AS fga,
      SUM(field_goals_made) AS fgm,
      SUM(free_throws_att) AS fta,
      SUM(free_throws_made) AS ftm,
      SUM(personal_fouls) AS pf
    FROM `bigquery-public-data.ncaa_basketball.mbb_players_games_sr`
    WHERE full_name IN UNNEST({player_names})
    GROUP BY player
    """
    return pandas_gbq.read_gbq(
        query_advanced, project_id=project_id, location="US"
    )


def generate_visualizations(merged_metrics):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=merged_metrics,
        x="college_bpm",
        y="box_plus_minus",
        hue="position",
        palette="Set2",
        alpha=0.8,
        s=70,
    )
    plt.title(
        "College Box Plus-Minus Proxy vs. NBA Box Plus-Minus (BPM)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("College BPM (Estimated Efficiency)")
    plt.ylabel("NBA Box Plus-Minus (BPM)")
    plt.legend(title="Position", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=merged_metrics,
        x="college_ws",
        y="win_shares",
        hue="position",
        palette="Set2",
        alpha=0.8,
        s=70,
    )
    plt.title(
        "College Win Shares Proxy vs. NBA Win Shares (WS)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("College Win Shares (Estimated)")
    plt.ylabel("NBA Cumulative Win Shares (WS)")
    plt.legend(title="Position", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def main():
    auth.authenticate_user()
    project_id = "project-2a43f60c-d493-4b1c-ae4"

    cleaned_draft_data = load_clean_nba_draft_data()
    player_names = cleaned_draft_data["player"].unique().tolist()

    college_raw_metrics = fetch_college_statistics(player_names, project_id)

    college_raw_metrics["college_bpm"] = (
        college_raw_metrics["pts"]
        + college_raw_metrics["reb"]
        + college_raw_metrics["ast"]
        + college_raw_metrics["stl"]
        + college_raw_metrics["blk"]
        - (college_raw_metrics["fga"] - college_raw_metrics["fgm"])
        - (college_raw_metrics["fta"] - college_raw_metrics["ftm"])
        - college_raw_metrics["tov"]
    ) / college_raw_metrics["games_played"]

    college_raw_metrics["college_ws"] = (
        college_raw_metrics["pts"]
        + 0.4 * college_raw_metrics["ftm"]
        - 0.7 * college_raw_metrics["fga"]
        - 0.4 * college_raw_metrics["fta"]
        + 0.7 * college_raw_metrics["reb"]
        + 0.7 * college_raw_metrics["ast"]
        + 1.0 * college_raw_metrics["stl"]
        + 1.0 * college_raw_metrics["blk"]
        - 1.0 * college_raw_metrics["tov"]
        - 0.4 * college_raw_metrics["pf"]
    ) / 10.0

    merged_metrics = pd.merge(
        cleaned_draft_data[
            ["player", "box_plus_minus", "win_shares", "position"]
        ],
        college_raw_metrics[["player", "college_bpm", "college_ws"]],
        on="player",
        how="inner",
    )

    generate_visualizations(merged_metrics)


if __name__ == "__main__":
    main()
