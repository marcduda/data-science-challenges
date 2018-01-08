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
print(len(df_booking['booking_user_id'].unique()))
df_booking =df_booking.rename(columns = {'arrial_datetime_2':'arrival_datetime_2'})
df_search=df_search.rename(columns = {'search_id':'booking_id','search_user_id':'booking_user_id','search_time':'booking_time'})
df_booking['booked']=1.0
df_search['booked']=0.0

df_booking=df_booking.set_index('booking_user_id')
df_search=df_search.set_index('booking_user_id')
df_all=pd.concat([df_booking,df_search],join='inner',ignore_index=True)#df_booking.join(df_search, on='booking_user_id', how='inner')
#
print(df_all.shape)
print(df_all['currency'].unique())
print(df_all['cabin_class'].unique())
print(df_all['passengers'].unique())
#%%
print(df_all['num_requests'].unique())
#print(np.where(booking_file['passengers']==0)[0].shape)
print(len(df_all['booking_id'].unique()))
ddd = df_all[df_all['booking_id'].isin(df_all['booking_id'].value_counts()[df_all['booking_id'].value_counts()>1].index)]
#print(df[df['search_id'].isin(df['search_id'].value_counts()[df['search_id'].value_counts()>1].index)])
print(df_all.shape[0]==len(df_all['booking_id'].unique()))
df_all = df_all[df_all.passengers != 0]
print(df_all['cabin_class'].unique())
df_all['cabin_class'] = df_all['cabin_class'].map({'economy':'economy', 'Economy':'economy', 'Business':'business', 'mixed':'mixed', 'First':'first', 'business':'business',
 'premium_economy':'premium_economy', 'first':'first', 'Premium Economy':'premium_economy', 'First Class':'first', 'COACH':'economy',
 'premiumEconomy':'premium_economy', 'ECONOMY':'economy'})
print(df_all['cabin_class'].unique())
print(len(df_all['booking_id'].unique()))
print(len(df_all['session_id'].unique()))
#%%
#by_session = df_all.groupby('session_id')

#%%
df_exrate = pd.read_csv('eurofxref.csv')
df_exrate.columns = [name.strip() for name in df_exrate.columns]
df_exrate.drop('',1)
print(df_exrate['USD'][0])
#%%
df_principal_currency = df_all[df_all.currency.isin(list(df_exrate.columns))]
sLength = len(df_principal_currency['currency'])
#df_principal_currency = df_principal_currency.assign(exchange=pd.Series(np.zeros(sLength)).values)
df_principal_currency['exchange']=df_principal_currency.apply(lambda df: df_exrate[df.currency][0], axis=1)
df_principal_currency['fare_eur']=df_principal_currency.fare/df_principal_currency.exchange#df_principal_currency.apply(lambda df: df.fare/df.exchange, axis=1)

#%%
def diffDays(df):
    return (df['departure_date'].date()-df['search_date'].date()).days

df = df_principal_currency
print(df['booking_time'][:10])
df['search_date']=df['booking_time'].apply(lambda x:date.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
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

df['distance_code'] = df.apply(codeDistance,axis=1)
print(df['distance_code'][:10])


#%%

df_feature = df[['booking_id','partner_id','passengers','num_requests','fare_eur','supplier','cabin_class',
                 'days_to_departure','search_day_of_year','search_day_of_week','departure_day_of_year','departure_day_of_week',
                 'days_to_return','return_day_of_year','return_day_of_week','has_return','nb_flights_departure','nb_flights_return',
                 'booked']]

df_feature.to_csv('features_classification.csv')
     
        