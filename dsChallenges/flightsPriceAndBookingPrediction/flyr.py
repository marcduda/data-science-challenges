#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 16:37:35 2017

@author: marcduda
"""
#Clean data

import pandas as pd
import numpy as np
import datetime as date
from pytz import timezone
import re
import warnings
warnings.filterwarnings('ignore')


df_search = pd.read_csv('search_data.csv')
df_booking = pd.read_csv('booking_data.csv')
df_search = df_search.drop_duplicates()
df_booking = df_booking.drop_duplicates()
df_booking=df_booking.rename(columns = {'booking_id':'search_id','booking_user_id':'search_user_id','booking_time':'search_time','arrial_datetime_2':'arrival_datetime_2'})
df_search = pd.concat([df_search,df_booking])
print(df_search['currency'].unique())
print(df_search['cabin_class'].unique())
print(df_search['passengers'].unique())
#%%
print(df_search['num_requests'].unique())
#print(np.where(booking_file['passengers']==0)[0].shape)
print(len(df_search['search_id'].unique()))
ddd = df_search[df_search['search_id'].isin(df_search['search_id'].value_counts()[df_search['search_id'].value_counts()>1].index)]
#print(df[df['search_id'].isin(df['search_id'].value_counts()[df['search_id'].value_counts()>1].index)])
df_search = df_search.drop_duplicates()
print(df_search.shape[0]==len(df_search['search_id'].unique()))
df_search = df_search[df_search.passengers != 0]
print(df_search['cabin_class'].unique())
df_search['cabin_class'] = df_search['cabin_class'].map({'economy':'economy', 'Economy':'economy', 'Business':'business', 'mixed':'mixed', 'First':'first', 'business':'business',
 'premium_economy':'premium_economy', 'first':'first', 'Premium Economy':'premium_economy', 'First Class':'first', 'COACH':'economy',
 'premiumEconomy':'premium_economy', 'ECONOMY':'economy'})
print(df_search['cabin_class'].unique())
#%%
all_dates = df_search['search_time'].unique().sort

df_exrate = pd.read_csv('eurofxref.csv')
#df_exrate = df_exrate.reset_index()
df_exrate.columns = [name.strip() for name in df_exrate.columns]
df_exrate.drop('',1)
#df_exrate.columns = [x for x in df_exrate.columns if x]
print(df_exrate['USD'][0])
#%%
df_principal_currency = df_search[df_search.currency.isin(list(df_exrate.columns))]
sLength = len(df_principal_currency['currency'])
#df_principal_currency = df_principal_currency.assign(exchange=pd.Series(np.zeros(sLength)).values)
df_principal_currency['exchange']=df_principal_currency.apply(lambda df: df_exrate[df.currency][0], axis=1)
df_principal_currency['fare_eur']=df_principal_currency.fare/df_principal_currency.exchange#df_principal_currency.apply(lambda df: df.fare/df.exchange, axis=1)

#%%
def diffDays(df):
    return (df['departure_date'].date()-df['search_date'].date()).days

import re
df = df_principal_currency
print(df['search_time'][:10])
df['search_date']=df['search_time'].apply(lambda x:date.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
df['departure_time'] = df['departure_datetime_1'].apply(lambda x:x.split(";")[0])
df['departure_date']=df['departure_time'].apply(lambda x:date.datetime.strptime(x[:19],"%Y-%m-%dT%H:%M:%S"))
df['days_to_departure']=df.apply(diffDays,axis=1)
#print(df[['search_date','departure_date','days_to_departure']][:10])
df['search_day_of_year']=df['search_date'].apply(lambda x:x.timetuple().tm_yday)
df['search_day_of_week']=df['search_date'].apply(lambda x:x.timetuple().tm_wday)
#print(df[['search_date','search_day_of_year']][:10])
df['departure_day_of_week']=df['departure_date'].apply(lambda x:x.timetuple().tm_wday)
df['departure_day_of_year']=df['departure_date'].apply(lambda x:x.timetuple().tm_yday)
#print(df[['search_date','search_day_of_year']][:10])
df['departure_day_of_week']=df['departure_date'].apply(lambda x:x.timetuple().tm_wday)

#%%
def splitPossibleNull(x):
    if x and str(x)!='nan': 
        return x.split(";")[0]
    else:
        return "2001-01-01T00:00:00"
def diffDaysReturn(df):
    if int(df['return_date'].year) >= int(df['search_date'].year):
        return (df['return_date'].date()-df['search_date'].date()).days
    else:
        #print(df['return_date'].year)
        return 0

df['return_time'] = df['arrival_datetime_1'].apply(splitPossibleNull)
df['return_date']=df['return_time'].apply(lambda x:date.datetime.strptime(x[:19],"%Y-%m-%dT%H:%M:%S"))
df['days_to_return']=df.apply(diffDaysReturn,axis=1)
print(df[['search_date','return_date','days_to_return']][:30])
df['return_day_of_year']=df['return_date'].apply(lambda x:x.timetuple().tm_yday)
df['return_day_of_week']=df['return_date'].apply(lambda x:x.timetuple().tm_wday)

df['has_return']=df['arrival_datetime_1'].apply(lambda x: float(str(x)!='nan'))
#%%
def numFlightsReturn(x):
    if x and str(x)!='nan': 
        return len(x.split(";"))
    else:
        return 0
    
df['nb_flights_departure']=df['flight_num_1'].apply(lambda x: len(x.split(';')))
df['nb_flights_return']=df['flight_num_2'].apply(numFlightsReturn)

#%%
def joinCoord(df_airport):
    return ";".join((str(df_airport['lat']),str(df_airport['lon']),str(df_airport['alt'])))

def codeDistance(df):
    airports = sorted([df['origin'],df['destination']], key=str.lower)
    return ''.join(airports)

df_airport = pd.read_csv('airports.csv',header=None, index_col=None)
print(df_airport[[4,6,7,8]].head(10))
df_airport = df_airport[[4,6,7,8]]
df_airport.columns = ['iata_code','lat','lon','alt']
#df_airport.set_index('iata_code', drop=True, inplace=True)
df_airport['coord']= df_airport.apply(joinCoord,axis=1)
df_airport.set_index('iata_code', drop=True, inplace=True)
df_airport = df_airport[['coord']]
print(df_airport['coord'][:10])#.head(10)
dictionary = df_airport.to_dict(orient="index")

#df['coord_origin']=df['origin'].apply(lambda x:dictionary[x])
#df['coord_destination']=df['destination'].apply(lambda x:dictionary[x])

df['distance_code'] = df.apply(codeDistance,axis=1)
print(df['distance_code'][:10])

#%%
print(len(df['search_id'].unique()))
print(len(df['search_user_id'].unique()))
print(len(df['session_id'].unique()))
by_user = df.groupby('search_user_id')
print(by_user.count()['search_id'][:20])

#%%
dddd = df.head(8)

df_feature = df[['search_id','partner_id','passengers','num_requests','fare_eur','supplier','cabin_class',
                 'days_to_departure','search_day_of_year','search_day_of_week','departure_day_of_year','departure_day_of_week',
                 'days_to_return','return_day_of_year','return_day_of_week','has_return','nb_flights_departure','nb_flights_return',
                 'distance_code']]

df_feature.to_csv('features_regression.csv')

#%%
#from pytz import country_timezones
#from pytz import timezone
#
#list_countries = [x for x in df['pos'].unique() if isinstance(x, str)]
#list_countries = [x for x in list_countries if len(x)==2]
#for country in list_countries:#df_search['POS'].unique()
#    country_tz = country_timezones(country)
#    if len(country_tz)>1:
#        print(country)
#        
        