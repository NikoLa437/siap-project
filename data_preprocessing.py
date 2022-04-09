import pandas as pd
import csv

RESULTS_DATASET_FILE_PATH = 'datasets/results.csv'
PLAYERS_DATASET_FILE_PATH = 'datasets/players.csv'

TEAMS_TO_NUM_FILE_PATH = 'datasets/team_to_num.csv'
PLAYERS_TO_NUM_FILE_PATH = 'datasets/player_to_num.csv'
MAPS_TO_NUM_FILE_PATH = 'datasets/map_to_num.csv'
COUTRY_TO_NUM_FILE_PATH= 'datasets/country_to_num.csv'
PLAYERS_WITH_COUNTRIES_FILE_PATH= 'datasets/player_with_country.csv'
FINAL_DATASET_FILE_PATH= 'datasets/final.csv'
FINAL_DATASET_WITH_COUNTRY_FILE_PATH= 'datasets/final_with_country.csv'
PLAYERS_AVG_RATING_FILE_PATH= 'datasets/players_avg_rating.csv'
PLAYER_WITH_COUNTRY_AVG_RATING = 'datasets/player_country_with_avg_rating.csv'


def save_dict_to_csv_file(data, file, header):
    try:
        with open(file, 'w', newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for key, value in data.items():
                writer.writerow([key, value])
    except IOError:
        print("I/O error")


def merge_country_and_avg_rating():
    players_avg_rating = pd.read_csv(PLAYERS_AVG_RATING_FILE_PATH,  encoding='utf-8')
    players_countries = pd.read_csv(PLAYERS_WITH_COUNTRIES_FILE_PATH,  encoding='utf-8')
    merged_data = pd.merge(players_avg_rating, players_countries, on='player_name')

    avg_players_rating_by_country = merged_data.groupby('country_number', as_index=False)['rating'].mean()
    final_merge = pd.merge(players_countries, avg_players_rating_by_country, on='country_number')

    #final_merge[['player_name', 'rating']].to_csv(PLAYER_WITH_COUNTRY_AVG_RATING, encoding='utf-8', index=False)
    #print(pd.read_csv(PLAYER_WITH_COUNTRY_AVG_RATING,  encoding='utf-8').set_index("player_name")['rating'].to_dict())

def get_country_percentage_in_dataset(count_countries_dict, player_num, country_num):
    return count_countries_dict[country_num]/player_num

def average_ranking_for_players():
    players_matches = pd.read_csv(PLAYERS_DATASET_FILE_PATH,  encoding='utf-8')
    avg_players_rating = players_matches.groupby('player_name', as_index=False)['rating'].mean()
    save_dict_to_csv_file(dict(avg_players_rating.values), PLAYERS_AVG_RATING_FILE_PATH, ['player_name', 'avg_rating'])

def convert_country_to_num():
    df = pd.read_csv(PLAYERS_DATASET_FILE_PATH,  encoding='utf-8')
    X = df[["country", 'player_name']]
    set_of_countries = set(list(X['country']))

    country_to_num = dict(zip(set_of_countries, range(len(set_of_countries))))

    player_with_country = {}
    for index, row in X.iterrows():
        player_with_country[row['player_name']] = country_to_num[row['country']]

    print(player_with_country)
    save_dict_to_csv_file(country_to_num, COUTRY_TO_NUM_FILE_PATH, ['country', 'country_number'])
    save_dict_to_csv_file(player_with_country, PLAYERS_WITH_COUNTRIES_FILE_PATH, ['player_name', 'country_number'])

    return player_with_country

def convert_team_to_num():
    df = pd.read_csv(RESULTS_DATASET_FILE_PATH,  encoding='utf-8')
    X = df[["team_1","team_2"]]
    set_of_teams = set(list(X['team_1']) + list(X['team_2']))

    team_to_num = dict(zip(set_of_teams, range(len(set_of_teams))))
    save_dict_to_csv_file(team_to_num, TEAMS_TO_NUM_FILE_PATH, ['team_name', 'team_number'])

    return team_to_num;

def convert_player_to_num():
    df = pd.read_csv(PLAYERS_DATASET_FILE_PATH,  encoding='utf-8')
    X = df[["player_name"]]
    set_of_players = set(list(X['player_name']))

    player_to_num = dict(zip(set_of_players, range(len(set_of_players))))
    save_dict_to_csv_file(player_to_num, PLAYERS_TO_NUM_FILE_PATH, ['player_name', 'player_number'])

    return player_to_num;

def convert_map_to_num():
    df = pd.read_csv(RESULTS_DATASET_FILE_PATH,  encoding='utf-8')
    X = df[["_map"]]
    set_of_maps = set(list(X['_map']))

    map_to_num = dict(zip(set_of_maps, range(len(set_of_maps))))
    save_dict_to_csv_file(map_to_num, MAPS_TO_NUM_FILE_PATH, ['map_name', 'map_number'])

    return map_to_num;

def data_set_processing():
    df = pd.read_csv(RESULTS_DATASET_FILE_PATH,  encoding='utf-8')
    dfPlayers = pd.read_csv(PLAYERS_DATASET_FILE_PATH,  encoding='utf-8')

    team_to_num = convert_team_to_num()
    player_to_num = convert_player_to_num()
    map_to_num = convert_map_to_num()
    country_to_num = convert_country_to_num()
    player_w_avg_country_rating = pd.read_csv(PLAYER_WITH_COUNTRY_AVG_RATING,  encoding='utf-8').set_index("player_name")['rating'].to_dict()

    grouped_by_match_id= df.groupby(['match_id'])
    dicts_array = []

    for group_name, df_group in grouped_by_match_id:
        dicts = {}
        number_of_map=1

        # iterate through maps 
        for row_index, row in df_group.iterrows():
            dicts['match_id'] = row["match_id"]
            dicts['team_1'] = team_to_num[row["team_1"]]
            dicts['team1_rank'] = row["rank_1"]
            dicts['team2_rank'] = row["rank_2"]
            dicts['team_2'] = team_to_num[row["team_2"]]
            dicts['map_' + str(number_of_map)] = map_to_num[row["_map"]]
            dicts['map_' + str(number_of_map) +'_winner'] = row["map_winner"]
            dicts['match_winner'] = row["match_winner"]
            number_of_map=number_of_map+1
        
        # dataframe of players on this match
        dfLists = dfPlayers.loc[dfPlayers['match_id']== group_name].groupby(['match_id'])

        hasPlayers=False # some match from results.csv haven't players data
        for group_name1, df_grou1p in dfLists:
            hasPlayers=True 
            team1num=1 # for dynamically generic string for players 
            team2num=1 

            #iterate through players in match
            for row_index, row in df_grou1p.iterrows():
                # some data in players.csv has some name for team1 and team2, because of that 
                # it isn't possible to diff players from team1 or team2
                if(team1num>6 or team2num>6): 
                    hasPlayers=False # set hasPlayers to false, then this data doesnt add to array
                    break;

                if(team_to_num[row['team']]==dicts['team_1']): # is this player in team1
                    dicts['player_' + str(team1num) + '_team_1_name'] = row['player_name']
                    dicts['player_' + str(team1num) + '_team_1_rating'] = row['rating']
                    dicts['player_' + str(team1num) + '_team_1'] = player_to_num[row['player_name']]
                    dicts['player_' + str(team1num) + '_team_1_country'] = country_to_num[row['player_name']]
                    dicts['player_' + str(team1num) + '_team_1_country_rating'] = player_w_avg_country_rating[row['player_name']]

                    team1num=team1num+1 # increase team1 players
                else: #  this player is in team1
                    dicts['player_' + str(team2num) + '_team_2_name'] = row['player_name']
                    dicts['player_' + str(team2num) + '_team_2_rating'] = row['rating']
                    dicts['player_' + str(team2num) + '_team_2'] = player_to_num[row['player_name']]
                    dicts['player_' + str(team2num) + '_team_2_country'] = country_to_num[row['player_name']]
                    dicts['player_' + str(team2num) + '_team_2_country_rating'] = player_w_avg_country_rating[row['player_name']]

                    team2num=team2num+1  # increase team1 players

            if(hasPlayers==False): # if we haven't valid player data skip this match
                break;

            #if team1num == 6 and team2num == 6 that means that for this match we have 10 players in dataset
            if(team1num+team2num != 12):
                hasPlayers= False
                break

        if(hasPlayers): # if we have valid data, add data to array
            dicts_array.append(dicts)

    df = pd.DataFrame.from_dict(dicts_array) 
    df.to_csv (FINAL_DATASET_WITH_COUNTRY_FILE_PATH, index = False, header=True)