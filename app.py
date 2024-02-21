# rsconnect deploy shiny ../shinyeintopfanalytics --name eintopfanalytics --title eintopfanalytics

from shiny import render, reactive
from shiny.express import input, ui

import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wordcloud
import datetime
import numpy as np

NOW = datetime.datetime.now()

PLOT_HEIGHT = 1400
PLOT_WIDTH = 800
SNS_FONT_SIZE = 0.7
HEATMAP_STEP_SIZE = 20
MAX_GROUP_NAME_LEN = 30

sns.set_theme(style="darkgrid")
sns.set(font="courier", font_scale=SNS_FONT_SIZE)


### Datenaufbereitung 
#@reactive.calc
#def filter_subevents(df):

# Daten scrapen
groups_raw = requests.get("https://eintopf.info/api/v1/groups").json()
events_raw = requests.get("https://eintopf.info/api/v1/events").json()

# Gruppen extrahieren
groups = [entry["name"] for entry in groups_raw]
groups_by_id = {entry["id"]: entry for entry in groups_raw}

parents = [event["parent"] for event in events_raw]
parents = [parent for parent in parents if parent != ""]

# Veranstaltungen extrahieren
events = []
for event in events_raw:
    for group in event["organizers"]:
        
        group = group.strip("id:")
        group = group if group not in groups_by_id.keys() else groups_by_id[group]["name"]
        group = group.strip()
        
        group_label = (
            group[0:MAX_GROUP_NAME_LEN - 3] + "..." 
            if len(group) > MAX_GROUP_NAME_LEN 
            else group
        )
        if len(group_label) < MAX_GROUP_NAME_LEN:
            group_label = " " * (MAX_GROUP_NAME_LEN - len(group_label)) + group_label

        events.append({
            "group": group,
            "group_label": group_label,
            "listed_group": True if group in groups else False,
            "event_name": event["name"],
            "event_datetime": event["start"],
            "event_description": event["description"],
            "subevent": True if parents.count(event["parent"]) >= 2 else False,
            "parent": event["parent"]
        })



df = pd.DataFrame(events)

df["event_datetime"] = pd.to_datetime(df.event_datetime)
df["event_date"] = df.event_datetime.apply(lambda x: x.date())

df["event_year"] = df.event_datetime.apply(lambda x: x.year)
df["event_year_str"] = df.event_year.apply(lambda x: str(x))

df["event_month"] = df.event_datetime.apply(lambda x: x.month)
df["event_month_str"] = df.event_month.apply(lambda x: str(x) if len(str(x)) == 2 else "0" + str(x))

df["event_year-month_str"] = df.event_year_str + "-" + df.event_month_str
df["event_yearmonth"] = df.event_year_str + df.event_month_str
df["event_yearmonth"] = df["event_yearmonth"].apply(lambda x: int(x))

df["counter"] = 1

df["past_month"] = True
for i, row in df.iterrows():
    if row.event_year >= NOW.year and row.event_month > NOW.month:
        df.loc[i, "past_month"] = False

df = (
    df
    .drop_duplicates(["event_name", "event_datetime"])
    .reset_index()
)



#if input.no_subevents():
#    df = df[df["subevent"] != True]
#return df
    
#return df


### App
ui.h1("Eintopf Analytics")
ui.hr()
ui.input_switch(
    "no_subevents", 
    "Keine regelmäßigen Veranstaltungen/Unterveranstaltungen anzeigen", 
    False,
    )
ui.h4("Monatl. Veranstaltungshäufigkeit im Zeitverlauf")



# Liniendiagramm: Veranstaltungshäufigkeit im Zeitverlauf
@render.plot
def freq_history():
    #df = filter_subevents(df)
    df = df.drop_duplicates(["event_name", "event_datetime"])

    df_listed = df.loc[df.listed_group==True,:]
    df_notlisted = df.loc[df.listed_group==False,:]

    def summonthly(df):
        return (
            df
            .groupby(by="event_year-month_str")
            ["counter"]
            .sum()
            .reset_index()
        )
        
    df_monthly_past = summonthly(df.loc[df["past_month"] == True])
    df_monthly_future = summonthly(df.loc[df["past_month"] == False])
    df_monthly_past_listed = summonthly(df_listed.loc[df_listed["past_month"] == True])
    df_monthly_future_listed = summonthly(df_listed.loc[df_listed["past_month"] == False])
    df_monthly_past_notlisted = summonthly(df_notlisted.loc[df_notlisted["past_month"] == True])
    df_monthly_future_notlisted = summonthly(df_notlisted.loc[df_notlisted["past_month"] == False])

    fig, ax = plt.subplots()

    if input.all_groups():
        sns.lineplot(
            data=df_monthly_past,
            x="event_year-month_str",
            y="counter",
            ax=ax,
            color="black",
        )
        sns.scatterplot(
            data=df_monthly_past,
            x="event_year-month_str",
            y="counter",
            ax=ax,
            color="black",
            label="Alle Gruppen",
        )
        if not input.history_only_past():
            sns.lineplot(
                data=df_monthly_future,
                x="event_year-month_str",
                y="counter",
                ax=ax,
                linestyle='--',
                color="black"
            )
            sns.scatterplot(
                data=df_monthly_future,
                x="event_year-month_str",
                y="counter",
                ax=ax,
                color="black"
            )

    if input.listed_groups():
        sns.lineplot(
            data=df_monthly_past_listed,
            x="event_year-month_str",
            y="counter",
            ax=ax,
            color="orange",
        )
        sns.scatterplot(
            data=df_monthly_past_listed,
            x="event_year-month_str",
            y="counter",
            ax=ax,
            color="orange",
            label="Gruppen mit Account",
        )
        if not input.history_only_past():
            sns.lineplot(
                data=df_monthly_future_listed,
                x="event_year-month_str",
                y="counter",
                ax=ax,
                linestyle='--',
                color="orange",
            )
            sns.scatterplot(
                data=df_monthly_future_listed,
                x="event_year-month_str",
                y="counter",
                ax=ax,
                color="orange",
            )

    if input.notlisted_groups():
        sns.lineplot(
            data=df_monthly_past_notlisted,
            x="event_year-month_str",
            y="counter",
            ax=ax,
            color="steelblue",
        )
        sns.scatterplot(
            data=df_monthly_past_notlisted,
            x="event_year-month_str",
            y="counter",
            ax=ax,
            color="steelblue",
            label="Gruppen mit Account",
        )
        if not input.history_only_past():
            sns.lineplot(
                data=df_monthly_future_notlisted,
                x="event_year-month_str",
                y="counter",
                ax=ax,
                linestyle='--',
                color="steelblue",
            )
            sns.scatterplot(
                data=df_monthly_future_notlisted,
                x="event_year-month_str",
                y="counter",
                ax=ax,
                color="steelblue",
            )
    ax.xaxis.set_major_locator(plt.MaxNLocator(16))
    plt.xticks(rotation=45)
    plt.xlabel("Monat")
    plt.ylabel("Anzahl Veranstaltungen")
    return fig

ui.input_switch("all_groups", "alle Gruppen", True)
ui.input_switch("listed_groups", "Gruppen mit Account", False)
ui.input_switch("notlisted_groups", "Gruppen ohne Account", False)
ui.input_switch("history_only_past", "nur Vergangenheit anzeigen", False)


# Balkendiagramm: Veranstaltungshäufigkeit nach Wochentag
ui.hr()
ui.h4("Veranstaltungshäufigkeit nach Wochentag")

@render.plot
def freq_by_day():
    df = filter_subevents(df)
    df = df.drop_duplicates(["event_name", "event_datetime"])

    
    
    df = df.loc[df["past_month"] == True]

    #df = df.loc[df["group"] != "Kesselbambule"]
    
    df["weekday"] = df["event_datetime"].apply(lambda x: x.weekday())
    df["year_sum"] = df.groupby("event_year").counter.transform(sum)
    df["counter"] = df["counter"] / df["year_sum"]
    df_day = df.groupby(["event_year_str", "weekday"])["counter"].sum().reset_index()
    
    fig, ax = plt.subplots()
    g = sns.barplot(
        data=df_day, 
        x="weekday", 
        y="counter", 
        hue= "event_year_str" if input.weekdays_per_year() else None, 
        ax=ax,
        )
    g.set_xticklabels(['Mo','Di','Mi','Do','Fr','Sa','So'])
    plt.xlabel("Wochentag")
    plt.ylabel("Anteil der Veranstaltungen")
    
    if input.weekdays_per_year():
        plt.legend(title="Jahr")
    
    return fig

ui.input_switch("weekdays_per_year", "pro Jahr anzeigen", False)


ui.hr()
ui.h4("Monatl. Veranstaltungshäufigkeit pro Gruppe")


def create_heatmap(i, show_xlabel=True, buffer=0):
    df = filter_subevents(df)
    
    if input.heatmap_only_past():
        df = df[df["past_month"] == True]

    df["total"] = df.groupby("group_label").counter.transform(sum)

    df_monthly = (
        df
        .groupby(by=["group_label", "event_year-month_str"])
        ["counter"]
        .sum()
        .reset_index()
    )

    df_monthly_pivot = (
        df_monthly
        .pivot(columns="event_year-month_str", index="group_label", values="counter")
        .fillna(0)
    )
    
    df["max_event_yearmonth"] = df.groupby(["group_label"]).event_yearmonth.transform(max)
    df["min_event_yearmonth"] = df.groupby(["group_label"]).event_yearmonth.transform(min)
    group_order = (
        df
        .groupby("group_label")
        [["max_event_yearmonth", "min_event_yearmonth"]]
        .mean()
        .reset_index()
        .sort_values(["max_event_yearmonth", "min_event_yearmonth"])
        .group_label
        .to_list()
    )

    df_monthly_pivot.index = pd.CategoricalIndex(
        df_monthly_pivot.index,
        categories=group_order,
        )
    df_monthly_pivot.sort_index(level=0, inplace=True, ascending=False)
    
    fig, ax = plt.subplots()
    fig.set_size_inches(20,20)
    g = sns.heatmap(
        data=df_monthly_pivot[i:i+HEATMAP_STEP_SIZE+buffer],
        linewidths=.5,
        cmap="Reds",
        cbar=False,
        ax=ax,
        vmax=4,
        vmin=0,

    )
    plt.xticks(rotation=90)
    xlabels = [t.get_text() for t in ax.get_xticklabels()]
    xlabels = [(xlabels[i] if i % 3 == 0 else None) for i in range(len(xlabels))]

    ax.set_xticklabels(xlabels)
    if show_xlabel:
        plt.xlabel("Monat", fontsize=SNS_FONT_SIZE)
    plt.ylabel("Gruppe (nur eingetragene)", fontsize=SNS_FONT_SIZE)
    return fig


@render.plot()
def heatmap():
    return create_heatmap(0 * HEATMAP_STEP_SIZE)

@render.plot()
def heatmap2():
    return create_heatmap(1 * HEATMAP_STEP_SIZE)

@render.plot()
def heatmap3():
    return create_heatmap(2 * HEATMAP_STEP_SIZE)

@render.plot()
def heatmap4():
    return create_heatmap(3 * HEATMAP_STEP_SIZE)

@render.plot()
def heatmap5():
    return create_heatmap(4 * HEATMAP_STEP_SIZE)

@render.plot()
def heatmap6():
    return create_heatmap(5 * HEATMAP_STEP_SIZE)

@render.plot()
def heatmap7():
    return create_heatmap(6 * HEATMAP_STEP_SIZE + 5)

ui.input_switch("heatmap_only_past", "nur Vergangenheit anzeigen", False)

ui.hr()
ui.h4("Die 50 Gruppen mit den meisten Veranstaltungen")

def create_barplot(i):
    df = filter_subevents(df)
    df = df["group_label"].value_counts().reset_index()

    max_freq = df.group_label.max()

    df = df[i:i+25]

    fig, ax = plt.subplots()
    
    sns.barplot(
        data=df,
        x="group_label",
        y="index",
        orient="h",
        ax=ax,
    )
    plt.ylabel("Gruppe")
    plt.xlabel("Veranstaltungshäufigkeit")
    plt.xlim(0,max_freq+5)
    return fig

@render.plot
def group_freq_barplot1():
    create_barplot(0)

@render.plot
def group_freq_barplot2():
    create_barplot(25)
