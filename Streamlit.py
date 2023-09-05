import streamlit as st
import pandas as pd


def get_data(league_name):
    if league_name == 'EPL':
        matches = pd.read_csv('Data23-22-21-EPL.csv', index_col=0)
    if league_name == 'LaLiga':
        matches = pd.read_csv('Data23-22-21-LaLiga.csv', index_col=0)
    if league_name == 'SerieA':
        matches = pd.read_csv('Data23-22-21-SerieA.csv', index_col=0)
    if league_name == 'BundesLiga':
        matches = pd.read_csv('Data23-22-21-Bundesliga.csv', index_col=0)
    if league_name == 'France1':
        matches = pd.read_csv('Data23-22-21-France.csv', index_col=0)

    return matches

class missingDict(dict):
    __missing__ = lambda self, key: key

map_values = {
        'RealMadrid': 'Real Madrid',
        'AtleticoMadrid': 'Atlético Madrid',
        'RealSociedad': 'Real Sociedad',
        'RealBetis': 'Betis',
        'AthleticClub': 'Athletic Club',
        'RayoVallecano': 'Rayo Vallecano',
        'CeltaVigo': 'Celta Vigo',
        'Cadiz': 'Cádiz',
        'Almeria': 'Almería',
        'Alaves': 'Alavés',
        'WestBromwichAlbion': 'West Brom',
        'SheffieldUnited': 'Sheffield Utd',
        'AstonVilla': 'Aston Villa',
        'CrystalPalace': 'Crystal Palace',
        'LeedsUnited': 'Leeds United',
        'LeicesterCity': 'Leicester City',
        'ManchesterCity': 'Manchester City',
        'NorwichCity': 'Norwich City',
        'BrightonandHoveAlbion': 'Brighton',
        'ManchesterUnited': 'Manchester Utd',
        'NewcastleUnited': 'Newcastle Utd',
        'TottenhamHotspur': 'Tottenham',
        'WestHamUnited': 'West Ham',
        'WolverhamptonWanderers': 'Wolves',
        'NottinghamForest': "Nott'ham Forest",
        'ParisSaintGermain': 'Paris S-G',
        'ClermontFoot': 'Clermont Foot',
        'Nimes': 'Nîmes',
        'SaintEtienne': 'Saint-Étienne',
        'BayerLeverkusen': 'Leverkusen',
        'BayernMunich': 'Bayern Munich',
        'EintrachtFrankfurt': 'Eint Frankfurt',
        'GreutherFurth': 'Greuther Fürth',
        'HerthaBSC': 'Hertha BSC',
        'Koln': 'Köln',
        'Mainz05': 'Mainz 05',
        'Monchengladbach': "M'Gladbach",
        'RBLeipzig': 'RB Leipzig',
        'Schalke04': 'Schalke 04',
        'UnionBerlin': 'Union Berlin',
        'WerderBremen': 'Werder Bremen',
        'HellasVerona': 'Hellas Verona',
        'Internazionale': 'Inter',
}
mapping = missingDict(**map_values)

def fix_team_name(data, mapping):
    data['Team'] = data['Team'].map(mapping)
    return data

def preprocess(matches):
    matches['Dist'] = matches['Dist'].fillna(matches['Dist'].mean())
    matches['FK'] = matches['FK'].fillna(matches['FK'].mean())

    matches['Date'] = pd.to_datetime(matches['Date'])
    matches['venue_code'] = matches['Venue'].astype('category').cat.codes
    matches['opp_code'] = matches['Opponent'].astype('category').cat.codes
    matches['team_code'] = matches['Team'].astype('category').cat.codes
    matches['hour'] = matches['Time'].str.replace(':.+', '', regex=True).astype('int')
    matches['day_code'] = matches['Date'].dt.dayofweek
    matches['target'] = (matches['Result'] == 'W').astype('int')
    matches['round_code'] = matches['Round'].str.replace('Matchweek ', '', regex=True).astype('int')

    return matches

cols = ['GF', 'GA', 'Sh', 'SoT', 'Dist', 'FK', 'PK', 'PKatt']
new_cols = [f'{c}_rolling' for c in cols]

def rolling_average(group, cols, new_cols):
    group = group.sort_values('Date')
    rolling_stats = group[cols].rolling(5, closed='left').mean()
    group['GF_rolling'] = rolling_stats['GF']
    group['GA_rolling'] = rolling_stats['GA']
    group['Sh_rolling'] = rolling_stats['Sh']
    group['SoT_rolling'] = rolling_stats['SoT']
    group['Dist_rolling'] = rolling_stats['Dist']
    group['FK_rolling'] = rolling_stats['FK']
    group['PK_rolling'] = rolling_stats['PK']
    group['PKatt_rolling'] = rolling_stats['PKatt']
    group[new_cols] = group[new_cols].fillna(group[~(group[new_cols].isna())][new_cols].mean())
    return group


st.title("Predict FootBall Match Result")
st.write("Please Select a League:")

leagues = ['EPL', 'BundesLiga', 'LaLiga', 'SerieA', 'France1']
select_league = st.selectbox('Select League Please:', leagues)
matches = get_data(select_league)
matches = fix_team_name(matches, mapping)
matches = preprocess(matches)
matches_rolling = matches.groupby('Team', group_keys=True).apply(lambda x: rolling_average(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('Team')
matches_rolling.index = range(matches_rolling.shape[0])
matches_rolling['match_id'] = matches_rolling[['Team', 'Opponent', 'Date']].apply(lambda x: ''.join(sorted([str(i) for i in x])), axis=1)
matches_rolling = matches_rolling.drop_duplicates(['match_id'])
predictors = ['team_code', 'venue_code', 'opp_code', 'hour', 'day_code', 'GF_rolling', 'GA_rolling', 'Sh_rolling', 'SoT_rolling', 'Dist_rolling', 'FK_rolling', 'PK_rolling', 'PKatt_rolling']



seasons = matches_rolling['Season'].unique()
selected_season = st.selectbox("Select a Season:", seasons)
filtered_season = matches_rolling[matches_rolling['Season'] == selected_season]


teams = filtered_season['Team'].unique()
selected_team = st.selectbox("Select Team:", teams)
df_team = filtered_season[(filtered_season['Team'] == selected_team) | (filtered_season['Opponent'] == selected_team)]



opponents = filtered_season['Opponent'].unique()
selected_opponent = st.selectbox("Select Opponent:", opponents)
df_match = df_team[(df_team['Opponent'] == selected_opponent) | (df_team['Team'] == selected_opponent)]
df_opp = filtered_season[(filtered_season['Team'] == selected_opponent) | (filtered_season['Opponent'] == selected_opponent)]

st.write(df_match[['Date', 'Day', 'Time', 'Venue', 'Round', 'Team', 'Opponent', 'Result']])


weeks = df_match['Round'].unique()
selected_week = st.selectbox("Select Match Week:", weeks)

# filtered_data_team = filtered_season[(filtered_season['Team'] == selected_team)]
# filtered_data_opp = filtered_season[(filtered_season['Team'] == selected_opponent)]

df_opp = df_opp.sort_values('Date')
df_team = df_team.sort_values('Date')

df_team.index = range(df_team.shape[0])
df_opp.index = range(df_opp.shape[0])

import re

before_week_team = df_team[df_team['round_code'] < int(re.findall("\d+", selected_week)[0])]
before_week_opp = df_opp[df_opp['round_code'] < int(re.findall("\d+", selected_week)[0])]

# st.write(before_week_team)


select_week_team = df_team[df_team['Round'] == selected_week]
select_week_opp = df_opp[df_opp['Round'] == selected_week]



df_team_rolling = before_week_team.tail(10)
# st.write(df_team_rolling)
df_opp_rolling = before_week_opp.tail(10)



if st.checkbox('Statistics for the Last 10 Games'):
    import matplotlib.pyplot as plt
    if st.button(f'{selected_team}'):
        st.write(df_team_rolling[['Team', 'GF', 'GA', 'Opponent', 'Result']])
        fig, ax = plt.subplots()
        ax.plot(df_team_rolling["GF"], label=f"{selected_team} Goals For ")
        ax.plot(df_team_rolling["GA"], label=f"{selected_team} Goals Against ")
        ax.set_xlabel("Games")
        ax.set_ylabel("Goals")
        ax.set_title("Statistics for the Last 10 Games")
        ax.legend()
        st.pyplot(fig)

        fig1, ax = plt.subplots()
        ax.plot(df_team_rolling["Sh"], label=f"{selected_team} Shoots")
        ax.set_xlabel("Games")
        ax.set_ylabel("Shoots")
        # ax.set_title("Statistics for the Last 10 Games")
        ax.legend()
        st.pyplot(fig1)

        fig2, ax = plt.subplots()
        ax.plot(df_team_rolling["xG"], label=f"{selected_team} Expected Goals For")
        ax.plot(df_team_rolling["xGA"], label=f"{selected_team} Expected Goals Against")
        ax.set_xlabel("Games")
        ax.set_ylabel("Expected")
        # ax.set_title("Statistics for the Last 10 Games")
        ax.legend()
        st.pyplot(fig2)

        st.write(select_week_team[['GF_rolling', 'GA_rolling', 'Sh_rolling', 'SoT_rolling', 'Dist_rolling', 'FK_rolling', 'PK_rolling', 'PKatt_rolling']])


    if st.button(f'{selected_opponent}'):
        st.write(df_opp_rolling[['Team', 'GF', 'GA', 'Opponent', 'Result']])
        fig, ax = plt.subplots()
        ax.plot(df_opp_rolling["GF"], label=f"{selected_opponent} Goals For ")
        ax.plot(df_opp_rolling["GA"], label=f"{selected_opponent} Goals Against ")
        # ax.set_xlabel("Games")
        ax.set_ylabel("Goals")
        ax.set_title("Statistics for the Last 10 Games")
        ax.legend()
        st.pyplot(fig)

        fig1, ax = plt.subplots()
        ax.plot(df_opp_rolling["Sh"], label=f"{selected_opponent} Shoots")
        # ax.set_xlabel("Games")
        ax.set_ylabel("Shoots")
        # ax.set_title("Statistics for the Last 10 Games")
        ax.legend()
        st.pyplot(fig1)

        fig2, ax = plt.subplots()
        ax.plot(df_opp_rolling["xG"], label=f"{selected_opponent} Expected Goals For")
        ax.plot(df_opp_rolling["xGA"], label=f"{selected_opponent} Expected Goals Against")
        ax.set_xlabel("Games")
        ax.set_ylabel("Expected")
        # ax.set_title("Statistics for the Last 10 Games")
        ax.legend()
        st.pyplot(fig2)

        st.write(select_week_team[['GF_rolling', 'GA_rolling', 'Sh_rolling', 'SoT_rolling', 'Dist_rolling', 'FK_rolling', 'PK_rolling', 'PKatt_rolling']])



train = df_team[df_team['Round'] != selected_week]
# train = filtered_data_team[filtered_data_team['Round'] != selected_week]
test = df_team[df_team['Round'] == selected_week]
# test = filtered_data_team[filtered_data_team['Round'] == selected_week]


def predict_vote(list, train, test):
    from sklearn.naive_bayes import MultinomialNB
    nb_model = MultinomialNB()
    nb_model.fit(train[predictors], train['target'])
    list = nb_model.predict(test[predictors])


    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=1000, min_samples_split=10, random_state=1)
    rf.fit(train[predictors], train['target'])
    list = rf.predict(test[predictors])


    # import xgboost as xgb
    # xgb_model = xgb.XGBClassifier(objective ='binary:logistic',
    #                               colsample_bytree = 0.3, learning_rate = 0.1,
    #                               max_depth = 5, alpha = 10, n_estimators = 1000)
    # xgb_model.fit(train[predictors], train['target'])
    # list = xgb_model.predict(test[predictors])


    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty='l2', tol=0.0001, C=100.0, random_state=1, max_iter=1000, n_jobs=-1)
    lr.fit(train[predictors], train['target'])
    list = lr.predict(test[predictors])


    from sklearn.svm import SVC
    svm = SVC(kernel='linear', tol=0.0001, C=100.0)
    svm.fit(train[predictors], train['target'])
    list = svm.predict(test[predictors])



    # from keras.models import Sequential
    # from keras.layers import Dense, Input
    # from keras.utils import to_categorical
    # from tensorflow import keras
    # import numpy as np
    #
    # model = Sequential()
    # model.add(Input(len(predictors)))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(2, activation='softmax'))
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    # model.fit(x_train,
    #           to_categorical(y_train, num_classes=2),
    #           validation_data=(x_valid, to_categorical(y_valid, num_classes=2)),
    #           epochs=30,
    #           batch_size=32)
    #
    # list = model.predict(x_test)
    # list = np.apply_along_axis(lambda x: np.argmax(x), axis=1, arr=list)



    from scipy import stats
    predictions = stats.mode(list, axis=0).mode.flatten()
    return predictions

if st.button("Predict Result!"):
    predictions = []
    predictions = predict_vote(predictions, train, test)
    if predictions == 1:
        st.write(f'{selected_team} Wins This Game.')
    else:
        st.write(f'{selected_team} Lose Or Draw This Game.')




