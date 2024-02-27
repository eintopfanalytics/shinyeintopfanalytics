# rsconnect deploy shiny ../shinyeintopfanalytics --name eintopfanalytics --title eintopfanalytics

from shiny.express import input, render, ui
from shinywidgets import render_plotly
import plotly.graph_objs as go

import plotly.express as px
import requests
import pandas as pd
import matplotlib.pyplot as plt
import wordcloud
import datetime
import numpy as np
from faicons import icon_svg as icon

# todos:
#X heatmap nach neuester veranstaltung sortieren 
# heatmap einklappbar machen
# Veranstaltungssteigerung pro Monat ausrechnen und anzeigen
# AUf Github verlinken
# Anzahl Gruppen im Zeitverlauf
# Gruppen mit Veranstaltung in letzten 12 Monaten
# Gruppen mit offensichtlich mehreren Namen zusammenführen
# Icons sinnvoll auswählen
# den bug mit zukunft/vergangenheit lösen

NOW = datetime.datetime.now()

PLOT_HEIGHT = 1400
PLOT_WIDTH = 800
SNS_FONT_SIZE = 0.7
HEATMAP_STEP_SIZE = 20
MAX_GROUP_NAME_LEN = 30

HEIGHT1 = 2000
FONT_SIZE1 = 9

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
        #if len(group_label) < MAX_GROUP_NAME_LEN:
        #    group_label = " " * (MAX_GROUP_NAME_LEN - len(group_label)) + group_label

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

df["future"] = False
for i, row in df.iterrows():
    if row.event_year >= NOW.year and row.event_month > NOW.month:
        df.loc[i, "future"] = True

df = (
    df
    .drop_duplicates(["event_name", "event_datetime"])
    .reset_index()
)

df_group_event = df.copy()

def filter_df(df):
    if input.drop_subevents():
        df = df[df["subevent"]!=True]
    
    if input.drop_kb():
        df = df[df["group"]!="Kesselbambule"]
    
    if input.drop_future():
        df = df[df["future"]!=True]
    return df



####################################################################################

### App

ui.page_opts(title = "Eintopf Analytics")

# Sidebar
with ui.sidebar():
    ui.h6("Globale Filter")
    ui.input_switch(
        "drop_subevents", 
        "Unterveranstaltungen ausschließen", 
        False,
        )
    ui.input_switch(
        "drop_kb", 
        "Kesselbambule ausschließen", 
        True,
        )
    ui.input_switch(
        "drop_future", 
        "Zukunft ausschließen", 
        True,
        )
    
    ui.hr()
    ui.h6("Info")
    ui.help_text("Die verwendeten Daten werden mit jedem Aufruf dieser Seite von hier geladen: eintopf.info/api/v1/events")


with ui.layout_column_wrap():

    with ui.value_box(showcase=icon("flag")):
        "Events"
        @render.express
        def n_events():
            df = df_group_event.copy()
            df = filter_df(df)
            str(df.counter.sum())

    with ui.value_box(showcase=icon("arrow-trend-up")):
        "Events im Monat"
        @render.express
        def n_events2():
            df = df_group_event.copy()
            df = filter_df(df)
            str(round(df.groupby("event_year-month_str").counter.sum().reset_index().counter.mean()))

    with ui.value_box(showcase=icon("arrow-trend-up")):
        "Events im Monat"
        @render.express
        def n_events3():
            df = df_group_event.copy()
            df = filter_df(df)
            str(round(df.groupby("event_year-month_str").counter.sum().reset_index().counter.mean()))


with ui.layout_column_wrap():
    with ui.value_box(showcase=icon("users")):
        "Gruppen"
        @render.express
        def n_groups():
            df = df_group_event.copy()
            df = filter_df(df)
            str(len(df.group.unique()))
    
    with ui.value_box(showcase=icon("users")):
        "Gruppen mit mind. 2 Events"
        @render.express
        def n_groups2():
            df = df_group_event.copy()
            df = filter_df(df)
            df = df.groupby("group").counter.sum().reset_index()
            df = df.loc[df.counter > 1]
            str(len(df.group.unique()))
    
    with ui.value_box(showcase=icon("users")):
        "Gruppen aktiv letzte 6 Monate"
        @render.express
        def n_groups3():
            df = df_group_event.copy()
            df = filter_df(df)
            n = 0
            for group in df.group.unique():
                df_group_temp = df.loc[df["group"]==group,:]
                for i, row in df_group_temp.iterrows():
                    if row["event_datetime"] >= datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=182):
                        n += 1
                        break
            str(n)

            

with ui.layout_column_wrap():
    # Liniendiagramm
    with ui.card(full_screen=True):
        ui.card_header("Monatl. Veranstaltungshäufigkeit im Zeitverlauf")
        
        
        @render_plotly
        def freq_history():
            # Datensatz aufbereiten
            df = df_group_event.copy()
            df = filter_df(df)
            df = df.drop_duplicates(["event_name", "event_datetime"])
            df_monthly = df.groupby(by="event_year-month_str")["counter"].sum().reset_index()
            df_monthly["event_year-month_str"] = pd.to_datetime(df_monthly["event_year-month_str"])
            
            if not input.show_trend_line():
                # Diagramm erstellen
                fig = px.line(
                    df_monthly,
                    x="event_year-month_str", 
                    y="counter",
                    labels={
                        "event_year-month_str": "Monat",
                        "counter": "Anzahl Veranstaltungen",
                        },
                    )
            else:
                fig = px.scatter(
                    df_monthly,
                    x="event_year-month_str", 
                    y="counter",
                    labels={
                        "event_year-month_str": "Monat",
                        "counter": "Anzahl Veranstaltungen",
                        },
                    trendline="lowess",
                    )
            #fig = px.scatter(df, x="date", y="GOOG", trendline="lowess")
            return fig
        ui.input_switch("show_trend_line", "Trend anzeigen", False)

    # Balkendiagramm: Veranstaltungshäufigkeit nach Wochentag
    with ui.card(full_screen=True):
        ui.card_header("Veranstaltungshäufigkeit nach Wochentag")

        @render_plotly
        def freq_by_day():
            df = df_group_event.copy()
            df = filter_df(df)
            df = df.drop_duplicates(["event_name", "event_datetime"])
            df["weekday"] = df["event_datetime"].apply(lambda x: x.weekday())

            labels={
                "weekday": "Wochentag",
                "perc": "% Veranstaltungen",
                "event_year_str": "Jahr",
            }

            if input.weekdays_per_year():
                df_year_day = df.copy()
                df_year_day["year_sum"] = df_year_day.groupby("event_year").counter.transform(sum)
                df_year_day["perc"] = (df_year_day["counter"] / df_year_day["year_sum"]) * 100
                df_year_day = df_year_day.groupby(["event_year_str", "weekday"])["perc"].sum().reset_index()
                fig = px.bar(data_frame=df_year_day, x="weekday", y="perc", color="event_year_str", barmode='group', labels=labels)
            
            else:
                df_day = df.copy()
                df_day = df_day.groupby(["weekday"])["counter"].sum().reset_index()
                df_day["perc"] = (df_day.counter / df_day.counter.sum()) * 100
                fig = px.bar(data_frame=df_day, x="weekday", y="perc", labels=labels)
            
            fig.update_xaxes(
                labelalias={
                    0: "Mo",
                    1: "Di",
                    2: "Mi",
                    3: "Do",
                    4: "Fr",
                    5: "Sa",
                    6: "So",
                },
            )
            return fig

        ui.input_switch("weekdays_per_year", "pro Jahr anzeigen", False)
        


with ui.layout_column_wrap():
    def create_heatmap():
        df = df_group_event.copy()
        df = filter_df(df)

        df["total"] = df.groupby("group_label").counter.transform(sum)

        

        df_monthly = (
            df
            .groupby(by=["group_label", "event_year-month_str"])
            ["counter"]
            .sum()
            .reset_index()
        )
        #df_monthly["counter"] = df_monthly["counter"].apply(lambda x: 4 if x >= 4 else x)
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

        df_event_date = df.groupby("group_label").event_datetime.max().reset_index()
        df_event_date = df_event_date.sort_values("event_datetime", ascending=True)
        group_order_date = df_event_date.group_label.to_list()

        df_monthly_pivot.index = pd.CategoricalIndex(
            df_monthly_pivot.index,
            categories=group_order_date,
            )
        df_monthly_pivot.sort_index(level=0, inplace=True, ascending=False)
        
        fig = px.imshow(
            df_monthly_pivot if (input.full_heatmap() or input.full_barplot()) else df_monthly_pivot[0:25],
            text_auto=False,
            color_continuous_scale='Greys',
            labels={
                "y": "Gruppe",
                "x": "Monat",
                "color": "Anzahl",
                },
            zmax=4,
            )
        fig.update_layout(
            height=HEIGHT1 if (input.full_heatmap() or input.full_barplot()) else None,
            coloraxis_showscale=False,
            yaxis = dict(
                tickfont = dict(size=FONT_SIZE1)),
            )
        return fig

    with ui.card(full_screen=True):
        ui.card_header("Monatl. Veranstaltungshäufigkeit pro Gruppe")
        @render_plotly()
        def heatmap():
            return create_heatmap()
        ui.input_switch("full_heatmap", "Alle Gruppen zeigen", False)
        #ui.card_footer("Info: Werte >= 4 werden in derselben Farbe dargestellt.")

    def create_barplot():
        df = df_group_event.copy()
        df = filter_df(df)
        df = df["group_label"].value_counts().reset_index()
        df = df.sort_values("group_label", ascending=True)

        max_freq = df.group_label.max()

        fig = px.bar(
            data_frame=df if (input.full_barplot() or input.full_heatmap()) else df[-25:], 
            x="group_label", 
            y="index", 
            orientation='h',
            labels={
                "group_label": "Anzahl",
                "index": "Gruppe",
                },
            )

        fig.update_layout(
            height=HEIGHT1 if (input.full_heatmap() or input.full_barplot()) else None,
            coloraxis_showscale=False,
            yaxis = dict(
                tickfont = dict(size=FONT_SIZE1)),
            )
        
        return fig

    with ui.card(full_screen=True):
        ui.card_header("Veranstaltungshäufigkeit nach Gruppe")
        @render_plotly()
        def group_freq_barplot1():
            return create_barplot()
        ui.input_switch("full_barplot", "Alle Gruppen anzeigen", False)