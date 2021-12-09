{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "heading_collapsed": true
   },
   "source": [
    "# Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:29:09.346812Z",
     "start_time": "2021-07-16T21:29:09.338834Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import pickle\n",
    "import time\n",
    "from custom_methods import preprocessing, calc_time\n",
    "\n",
    "datapath = '../Data/'\n",
    "\n",
    "# start timer\n",
    "startTime = time.time()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "heading_collapsed": true
   },
   "source": [
    "# Billing Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:29:20.723516Z",
     "start_time": "2021-07-16T21:29:09.489430Z"
    },
    "hidden": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Length: 3821082\n",
      "Accounts: 98054\n",
      "Premises: 64815\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ARREARSMONTH</th>\n",
       "      <th>RES_EL_CUR120_DAYS</th>\n",
       "      <th>RES_EL_CUR22_DAYS</th>\n",
       "      <th>RES_EL_CUR30_DAYS</th>\n",
       "      <th>RES_EL_CUR60_DAYS</th>\n",
       "      <th>RES_EL_CUR90_DAYS</th>\n",
       "      <th>RES_EL_CUR_BAL_AMT</th>\n",
       "      <th>RES_EL_OVER_120_DAYS</th>\n",
       "      <th>RES_GAS_CUR120_DAYS</th>\n",
       "      <th>RES_GAS_CUR22_DAYS</th>\n",
       "      <th>...</th>\n",
       "      <th>SEVERANCE_ELECTRIC</th>\n",
       "      <th>SEVERANCE_GAS</th>\n",
       "      <th>MONTHID</th>\n",
       "      <th>CITY_TOT_DUE</th>\n",
       "      <th>CITY_30_DAYS_PAST_DUE_AMT</th>\n",
       "      <th>CITY_60_DAYS_PAST_DUE_AMT</th>\n",
       "      <th>CITY_90_DAYS_PAST_DUE_AMT</th>\n",
       "      <th>SPA_PREM_ID</th>\n",
       "      <th>SPA_ACCT_ID</th>\n",
       "      <th>COVID_REMINDER</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>201512</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>90.02</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.0</td>\n",
       "      <td>90.02</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>4.885320e+10</td>\n",
       "      <td>131.59</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>139.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>201512</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>72.37</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.0</td>\n",
       "      <td>72.37</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>8.903202e+09</td>\n",
       "      <td>186.60</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>33.0</td>\n",
       "      <td>181.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>201512</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>528.84</td>\n",
       "      <td>81.34</td>\n",
       "      <td>0.0</td>\n",
       "      <td>610.18</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>9.672015e+08</td>\n",
       "      <td>331.86</td>\n",
       "      <td>130.83</td>\n",
       "      <td>72.7</td>\n",
       "      <td>0.0</td>\n",
       "      <td>37.0</td>\n",
       "      <td>17.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>201512</td>\n",
       "      <td>0.0</td>\n",
       "      <td>54.0</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.0</td>\n",
       "      <td>54.00</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>80.0</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>4.180520e+10</td>\n",
       "      <td>105.81</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>73.0</td>\n",
       "      <td>173.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>201512</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>5.075220e+10</td>\n",
       "      <td>98.11</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>110.0</td>\n",
       "      <td>140.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 32 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   ARREARSMONTH  RES_EL_CUR120_DAYS  RES_EL_CUR22_DAYS  RES_EL_CUR30_DAYS  \\\n",
       "0        201512                 0.0                0.0              90.02   \n",
       "1        201512                 0.0                0.0              72.37   \n",
       "2        201512                 0.0                0.0             528.84   \n",
       "3        201512                 0.0               54.0               0.00   \n",
       "4        201512                 0.0                0.0               0.00   \n",
       "\n",
       "   RES_EL_CUR60_DAYS  RES_EL_CUR90_DAYS  RES_EL_CUR_BAL_AMT  \\\n",
       "0               0.00                0.0               90.02   \n",
       "1               0.00                0.0               72.37   \n",
       "2              81.34                0.0              610.18   \n",
       "3               0.00                0.0               54.00   \n",
       "4               0.00                0.0                0.00   \n",
       "\n",
       "   RES_EL_OVER_120_DAYS  RES_GAS_CUR120_DAYS  RES_GAS_CUR22_DAYS  ...  \\\n",
       "0                   0.0                  0.0                 0.0  ...   \n",
       "1                   0.0                  0.0                 0.0  ...   \n",
       "2                   0.0                  0.0                 0.0  ...   \n",
       "3                   0.0                  0.0                80.0  ...   \n",
       "4                   0.0                  0.0                 0.0  ...   \n",
       "\n",
       "   SEVERANCE_ELECTRIC  SEVERANCE_GAS       MONTHID  CITY_TOT_DUE  \\\n",
       "0                 1.0            0.0  4.885320e+10        131.59   \n",
       "1                 NaN            NaN  8.903202e+09        186.60   \n",
       "2                 1.0            0.0  9.672015e+08        331.86   \n",
       "3                 NaN            NaN  4.180520e+10        105.81   \n",
       "4                 NaN            NaN  5.075220e+10         98.11   \n",
       "\n",
       "   CITY_30_DAYS_PAST_DUE_AMT  CITY_60_DAYS_PAST_DUE_AMT  \\\n",
       "0                       0.00                        0.0   \n",
       "1                       0.00                        0.0   \n",
       "2                     130.83                       72.7   \n",
       "3                       0.00                        0.0   \n",
       "4                       0.00                        0.0   \n",
       "\n",
       "   CITY_90_DAYS_PAST_DUE_AMT  SPA_PREM_ID  SPA_ACCT_ID  COVID_REMINDER  \n",
       "0                        0.0          3.0        139.0             NaN  \n",
       "1                        0.0         33.0        181.0             NaN  \n",
       "2                        0.0         37.0         17.0             NaN  \n",
       "3                        0.0         73.0        173.0             NaN  \n",
       "4                        0.0        110.0        140.0             NaN  \n",
       "\n",
       "[5 rows x 32 columns]"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Load Billing Data\n",
    "# This just grabs all the seperate billing data files\n",
    "fileyears = ['2015', '2016', '2017', '2018', '2019', '2020']\n",
    "path = 'SpaData_'\n",
    "df = pd.read_csv(datapath+path+fileyears[0]+'_Anon.csv')\n",
    "for fileyear in fileyears[1:]:\n",
    "    df = df.append(pd.read_csv(datapath+path+fileyear+'_Anon.csv'))\n",
    "\n",
    "rows = len(df)\n",
    "accts = df.SPA_ACCT_ID.nunique()\n",
    "premises = df.SPA_PREM_ID.nunique()\n",
    "print(f'Length: {rows}')\n",
    "print(f'Accounts: {accts}')\n",
    "print(f'Premises: {premises}')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:29:20.738462Z",
     "start_time": "2021-07-16T21:29:20.726505Z"
    },
    "hidden": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['ARREARSMONTH',\n",
       " 'RES_EL_CUR120_DAYS',\n",
       " 'RES_EL_CUR22_DAYS',\n",
       " 'RES_EL_CUR30_DAYS',\n",
       " 'RES_EL_CUR60_DAYS',\n",
       " 'RES_EL_CUR90_DAYS',\n",
       " 'RES_EL_CUR_BAL_AMT',\n",
       " 'RES_EL_OVER_120_DAYS',\n",
       " 'RES_GAS_CUR120_DAYS',\n",
       " 'RES_GAS_CUR22_DAYS',\n",
       " 'RES_GAS_CUR30_DAYS',\n",
       " 'RES_GAS_CUR60_DAYS',\n",
       " 'RES_GAS_CUR90_DAYS',\n",
       " 'RES_GAS_CUR_BAL_AMT',\n",
       " 'RES_GAS_OVER_120_DAYS',\n",
       " 'BREAK_ARRANGEMENT',\n",
       " 'BREAK_PAY_PLAN',\n",
       " 'CALL_OUT',\n",
       " 'CALL_OUT_MANUAL',\n",
       " 'DUE_DATE',\n",
       " 'FINAL_NOTICE',\n",
       " 'PAST_DUE',\n",
       " 'SEVERANCE_ELECTRIC',\n",
       " 'SEVERANCE_GAS',\n",
       " 'MONTHID',\n",
       " 'CITY_TOT_DUE',\n",
       " 'CITY_30_DAYS_PAST_DUE_AMT',\n",
       " 'CITY_60_DAYS_PAST_DUE_AMT',\n",
       " 'CITY_90_DAYS_PAST_DUE_AMT',\n",
       " 'SPA_PREM_ID',\n",
       " 'SPA_ACCT_ID',\n",
       " 'COVID_REMINDER']"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns.to_list()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "hidden": true
   },
   "source": [
    "## Rename Attributes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:29:22.011382Z",
     "start_time": "2021-07-16T21:29:20.740456Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "df = df.rename({'ARREARSMONTH':'MONTH'}, axis=1)\n",
    "df = df.drop(['MONTHID', 'COVID_REMINDER'], axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "hidden": true
   },
   "source": [
    "## Reformat Dates"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:29:22.042190Z",
     "start_time": "2021-07-16T21:29:22.014379Z"
    },
    "hidden": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Earliest Month: 201512\n",
      "Latest Month: 202012\n"
     ]
    }
   ],
   "source": [
    "print(f'Earliest Month: {df.MONTH.min()}')\n",
    "print(f'Latest Month: {df.MONTH.max()}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "hidden": true
   },
   "source": [
    "Use December, 2015 as month 0 - this is the earliest month in the billing data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:29:33.165427Z",
     "start_time": "2021-07-16T21:29:22.044187Z"
    },
    "hidden": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Earliest Month: 0\n",
      "Latest Month: 60\n"
     ]
    }
   ],
   "source": [
    "df.MONTH = df.MONTH.apply(lambda x: preprocessing.date_map(date=x, relative_to=201512, format='yyyymm'))\n",
    "print(f'Earliest Month: {df.MONTH.min()}')\n",
    "print(f'Latest Month: {df.MONTH.max()}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "hidden": true
   },
   "source": [
    "## Prepare for Matching\n",
    "Want to match on unique combinations of ('SPA_ACCT_ID', 'SPA_PREM_ID', 'MONTH')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:29:35.897224Z",
     "start_time": "2021-07-16T21:29:33.167473Z"
    },
    "hidden": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1    3799661\n",
       "2      10691\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = df.drop_duplicates()\n",
    "\n",
    "df.groupby(['SPA_ACCT_ID', 'SPA_PREM_ID', 'MONTH']).size().value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:29:36.150226Z",
     "start_time": "2021-07-16T21:29:35.900219Z"
    },
    "hidden": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "MONTH                              0\n",
       "RES_EL_CUR120_DAYS                 0\n",
       "RES_EL_CUR22_DAYS                  0\n",
       "RES_EL_CUR30_DAYS                  0\n",
       "RES_EL_CUR60_DAYS                  0\n",
       "RES_EL_CUR90_DAYS                  0\n",
       "RES_EL_CUR_BAL_AMT                 0\n",
       "RES_EL_OVER_120_DAYS               0\n",
       "RES_GAS_CUR120_DAYS                0\n",
       "RES_GAS_CUR22_DAYS                 0\n",
       "RES_GAS_CUR30_DAYS                 0\n",
       "RES_GAS_CUR60_DAYS                 0\n",
       "RES_GAS_CUR90_DAYS                 0\n",
       "RES_GAS_CUR_BAL_AMT                0\n",
       "RES_GAS_OVER_120_DAYS              0\n",
       "BREAK_ARRANGEMENT            3126575\n",
       "BREAK_PAY_PLAN               3126575\n",
       "CALL_OUT                     3126575\n",
       "CALL_OUT_MANUAL              3126575\n",
       "DUE_DATE                     3126575\n",
       "FINAL_NOTICE                 3126575\n",
       "PAST_DUE                     3126575\n",
       "SEVERANCE_ELECTRIC           3126575\n",
       "SEVERANCE_GAS                3126575\n",
       "CITY_TOT_DUE                  130815\n",
       "CITY_30_DAYS_PAST_DUE_AMT     130815\n",
       "CITY_60_DAYS_PAST_DUE_AMT     130815\n",
       "CITY_90_DAYS_PAST_DUE_AMT     130815\n",
       "SPA_PREM_ID                        0\n",
       "SPA_ACCT_ID                        0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:29:46.184092Z",
     "start_time": "2021-07-16T21:29:36.153272Z"
    },
    "hidden": true,
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original length: 3821082\n",
      "Original accounts: 98054\n",
      "\n",
      "New length: 3679537\n",
      "New accounts: 96840\n",
      "\n",
      "141545 Rows lost\n",
      "1214 Accounts lost\n",
      "\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "1    3679537\n",
       "dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = df[~df.CITY_TOT_DUE.isna()]\n",
    "\n",
    "# Change NA for the following attributes to 0\n",
    "# Assume no data means there have been 0 occurrances of each of these\n",
    "set_to_zero = [\n",
    "    'BREAK_ARRANGEMENT',\n",
    "    'BREAK_PAY_PLAN',\n",
    "    'CALL_OUT',\n",
    "    'CALL_OUT_MANUAL',\n",
    "    'DUE_DATE',\n",
    "    'FINAL_NOTICE',\n",
    "    'PAST_DUE',\n",
    "    'SEVERANCE_ELECTRIC',\n",
    "    'SEVERANCE_GAS',\n",
    "]\n",
    "\n",
    "for col in set_to_zero:\n",
    "    df[col] = df[col].replace(to_replace=np.nan, value=0)\n",
    "\n",
    "# Just choose last of duplicates\n",
    "print(f'Original length: {rows}')\n",
    "print(f'Original accounts: {accts}')\n",
    "print()\n",
    "df = df.groupby(['SPA_ACCT_ID', 'SPA_PREM_ID', 'MONTH']).last().reset_index()\n",
    "print(f'New length: {len(df)}')\n",
    "print(f'New accounts: {df.SPA_ACCT_ID.nunique()}')\n",
    "print(f'\\n{rows-len(df)} Rows lost')\n",
    "print(f'{accts-df.SPA_ACCT_ID.nunique()} Accounts lost\\n')\n",
    "\n",
    "df.groupby(['SPA_ACCT_ID', 'SPA_PREM_ID', 'MONTH']).size().value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "heading_collapsed": true
   },
   "source": [
    "# Service Agreements"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:29:47.421325Z",
     "start_time": "2021-07-16T21:29:46.187146Z"
    },
    "hidden": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Rows: 708465\n",
      "Accounts: 271001\n",
      "People: 305220\n",
      "Positive Cases: 2387\n",
      "\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>SPA_PREM_ID</th>\n",
       "      <th>SPA_ACCT_ID</th>\n",
       "      <th>spa_sa_id</th>\n",
       "      <th>SPA_PER_ID</th>\n",
       "      <th>ACCT_REL_TYPE_CD</th>\n",
       "      <th>CMIS_MATCH</th>\n",
       "      <th>START_DT</th>\n",
       "      <th>END_DT</th>\n",
       "      <th>SA_TYPE_DESCR</th>\n",
       "      <th>Class</th>\n",
       "      <th>APARTMENT</th>\n",
       "      <th>ENROLL_DATE</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>71748.0</td>\n",
       "      <td>238198.0</td>\n",
       "      <td>475493.0</td>\n",
       "      <td>70257.0</td>\n",
       "      <td>MAIN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2019-06-26</td>\n",
       "      <td>2020-06-30</td>\n",
       "      <td>Residential Electric WA</td>\n",
       "      <td>RESIDENTIAL</td>\n",
       "      <td>False</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>133474.0</td>\n",
       "      <td>123629.0</td>\n",
       "      <td>247488.0</td>\n",
       "      <td>163464.0</td>\n",
       "      <td>COTENANT</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2019-06-15</td>\n",
       "      <td>2020-06-30</td>\n",
       "      <td>Residential Electric WA</td>\n",
       "      <td>RESIDENTIAL</td>\n",
       "      <td>True</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>130677.0</td>\n",
       "      <td>24642.0</td>\n",
       "      <td>49442.0</td>\n",
       "      <td>239032.0</td>\n",
       "      <td>MAIN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2019-06-19</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Residential Electric WA</td>\n",
       "      <td>RESIDENTIAL</td>\n",
       "      <td>True</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>133474.0</td>\n",
       "      <td>123629.0</td>\n",
       "      <td>247488.0</td>\n",
       "      <td>47487.0</td>\n",
       "      <td>MAIN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2019-06-15</td>\n",
       "      <td>2020-06-30</td>\n",
       "      <td>Residential Electric WA</td>\n",
       "      <td>RESIDENTIAL</td>\n",
       "      <td>True</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>49126.0</td>\n",
       "      <td>107286.0</td>\n",
       "      <td>214606.0</td>\n",
       "      <td>264436.0</td>\n",
       "      <td>MAIN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2019-06-18</td>\n",
       "      <td>2019-11-04</td>\n",
       "      <td>Residential Electric WA</td>\n",
       "      <td>RESIDENTIAL</td>\n",
       "      <td>True</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   SPA_PREM_ID  SPA_ACCT_ID  spa_sa_id  SPA_PER_ID ACCT_REL_TYPE_CD  \\\n",
       "0      71748.0     238198.0   475493.0     70257.0             MAIN   \n",
       "1     133474.0     123629.0   247488.0    163464.0         COTENANT   \n",
       "2     130677.0      24642.0    49442.0    239032.0             MAIN   \n",
       "3     133474.0     123629.0   247488.0     47487.0             MAIN   \n",
       "4      49126.0     107286.0   214606.0    264436.0             MAIN   \n",
       "\n",
       "  CMIS_MATCH    START_DT      END_DT            SA_TYPE_DESCR        Class  \\\n",
       "0        NaN  2019-06-26  2020-06-30  Residential Electric WA  RESIDENTIAL   \n",
       "1        NaN  2019-06-15  2020-06-30  Residential Electric WA  RESIDENTIAL   \n",
       "2        NaN  2019-06-19         NaN  Residential Electric WA  RESIDENTIAL   \n",
       "3        NaN  2019-06-15  2020-06-30  Residential Electric WA  RESIDENTIAL   \n",
       "4        NaN  2019-06-18  2019-11-04  Residential Electric WA  RESIDENTIAL   \n",
       "\n",
       "   APARTMENT ENROLL_DATE  \n",
       "0      False         NaN  \n",
       "1       True         NaN  \n",
       "2       True         NaN  \n",
       "3       True         NaN  \n",
       "4       True         NaN  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sa = pd.read_csv(datapath+'ServiceAgreements_Anon.csv').\\\n",
    "    rename({'spa_prem_id':'SPA_PREM_ID', 'spa_acct_id':'SPA_ACCT_ID', 'spa_per_id':'SPA_PER_ID', 'homelessMatch':'CMIS_MATCH', 'EnrollDate':'ENROLL_DATE', 'apartment':'APARTMENT'}, axis=1)\n",
    "sa_rows = len(sa)\n",
    "sa_ppl = sa.SPA_PER_ID.nunique()\n",
    "sa_accts = sa.SPA_ACCT_ID.nunique()\n",
    "sa_pos_ppl = sa[sa.CMIS_MATCH == True].SPA_PER_ID.nunique()\n",
    "\n",
    "print(f'Rows: {sa_rows}')\n",
    "print(f'Accounts: {sa_accts}')\n",
    "print(f'People: {sa_ppl}')\n",
    "print(f'Positive Cases: {sa_pos_ppl}\\n')\n",
    "sa.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "hidden": true
   },
   "source": [
    "## Transform\n",
    "Problems:\n",
    "* Some accounts have multiple people associated with them at a time, some only have one\n",
    "* Some people are associated with multiple accounts (sometimes at different 'ACCT_REL_TYPE_CD')  \n",
    "\n",
    "Solution:  \n",
    "* Only retain the main account holder for each account"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:29:53.323771Z",
     "start_time": "2021-07-16T21:29:47.424316Z"
    },
    "hidden": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Null Enroll Dates for P Cases: 0\n",
      "Positive Cases: 1935\n",
      "\n",
      "Grouping:\n",
      "1    270848\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# Convert Dates to months since December, 2015\n",
    "sa.ENROLL_DATE = sa.ENROLL_DATE.apply(lambda x: preprocessing.date_map(date=x, relative_to='2015-01-01', format='yyyy-mm-dd'))\n",
    "\n",
    "# Replace NaN with False in CMIS_MATCH\n",
    "sa.CMIS_MATCH = sa.CMIS_MATCH.replace(to_replace=np.nan, value=False).astype('bool')\n",
    "\n",
    "# Any null enroll dates for cmis_match? No - good\n",
    "print(f'Null Enroll Dates for P Cases: {sa[sa.CMIS_MATCH][\"ENROLL_DATE\"].isnull().sum()}')\n",
    "\n",
    "# Retain only columns we want to add to billing - note: all CMIS_MATCHes have ENROLL_DATEs\n",
    "sa.drop(['spa_sa_id', 'START_DT', 'END_DT', 'SA_TYPE_DESCR', 'Class'], axis=1, inplace=True)\n",
    "\n",
    "# Create list of accounts that have a cotenant\n",
    "cotenant_accounts = sa[sa['ACCT_REL_TYPE_CD'] == 'COTENANT']['SPA_ACCT_ID'].values\n",
    "# Only keep info regarding the 'MAIN' account holder\n",
    "sa = sa[sa['ACCT_REL_TYPE_CD'] == 'MAIN'].drop('ACCT_REL_TYPE_CD', axis=1)\n",
    "# Add boolean column for cotenant\n",
    "sa['HAS_COTENANT'] = sa['SPA_ACCT_ID'].isin(cotenant_accounts).astype('bool')\n",
    "del cotenant_accounts\n",
    "sa.drop_duplicates(inplace=True)\n",
    "\n",
    "# Group Enroll Dates into list\n",
    "enroll_dates = sa[~sa[\"ENROLL_DATE\"].isnull()].groupby([\"SPA_ACCT_ID\", \"SPA_PREM_ID\"])[\"ENROLL_DATE\"].unique()\n",
    "sa = sa.set_index(['SPA_ACCT_ID','SPA_PREM_ID']).sort_index()\n",
    "sa.update(enroll_dates)\n",
    "del enroll_dates\n",
    "sa[\"ENROLL_DATE\"] = sa[\"ENROLL_DATE\"].apply(lambda x: tuple([]) if np.isnan(x).all() else tuple(x))\n",
    "\n",
    "# If any CMIS_MATCH for person, then CMIS_MATCH for all instances of person\n",
    "sa.update(sa.groupby(['SPA_ACCT_ID', 'SPA_PREM_ID'])[\"CMIS_MATCH\"].any())\n",
    "sa.drop_duplicates(inplace=True)\n",
    "sa.reset_index(inplace=True)\n",
    "\n",
    "print(f'Positive Cases: {sa[sa.CMIS_MATCH].SPA_PER_ID.nunique()}')\n",
    "\n",
    "# Check Matching\n",
    "print('\\nGrouping:')\n",
    "print(sa.groupby(['SPA_ACCT_ID', 'SPA_PREM_ID']).size().value_counts())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "hidden": true
   },
   "source": [
    "## Join to df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:29:56.592655Z",
     "start_time": "2021-07-16T21:29:53.325767Z"
    },
    "hidden": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "SPA_ACCT_ID                       0\n",
       "SPA_PREM_ID                       0\n",
       "MONTH                             0\n",
       "RES_EL_CUR120_DAYS                0\n",
       "RES_EL_CUR22_DAYS                 0\n",
       "RES_EL_CUR30_DAYS                 0\n",
       "RES_EL_CUR60_DAYS                 0\n",
       "RES_EL_CUR90_DAYS                 0\n",
       "RES_EL_CUR_BAL_AMT                0\n",
       "RES_EL_OVER_120_DAYS              0\n",
       "RES_GAS_CUR120_DAYS               0\n",
       "RES_GAS_CUR22_DAYS                0\n",
       "RES_GAS_CUR30_DAYS                0\n",
       "RES_GAS_CUR60_DAYS                0\n",
       "RES_GAS_CUR90_DAYS                0\n",
       "RES_GAS_CUR_BAL_AMT               0\n",
       "RES_GAS_OVER_120_DAYS             0\n",
       "BREAK_ARRANGEMENT                 0\n",
       "BREAK_PAY_PLAN                    0\n",
       "CALL_OUT                          0\n",
       "CALL_OUT_MANUAL                   0\n",
       "DUE_DATE                          0\n",
       "FINAL_NOTICE                      0\n",
       "PAST_DUE                          0\n",
       "SEVERANCE_ELECTRIC                0\n",
       "SEVERANCE_GAS                     0\n",
       "CITY_TOT_DUE                      0\n",
       "CITY_30_DAYS_PAST_DUE_AMT         0\n",
       "CITY_60_DAYS_PAST_DUE_AMT         0\n",
       "CITY_90_DAYS_PAST_DUE_AMT         0\n",
       "SPA_PER_ID                   436786\n",
       "CMIS_MATCH                   436786\n",
       "APARTMENT                    436786\n",
       "ENROLL_DATE                  436786\n",
       "HAS_COTENANT                 436786\n",
       "dtype: int64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sa = sa.set_index(['SPA_ACCT_ID', 'SPA_PREM_ID'])\n",
    "df = df.join(sa, on=['SPA_ACCT_ID', 'SPA_PREM_ID'], how='left')\n",
    "\n",
    "del sa\n",
    "\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "hidden": true
   },
   "source": [
    "## Only Keep Known SPA_PER_ID's"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:29:58.763094Z",
     "start_time": "2021-07-16T21:29:56.595052Z"
    },
    "hidden": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "SPA_ACCT_ID                  0\n",
       "SPA_PREM_ID                  0\n",
       "MONTH                        0\n",
       "RES_EL_CUR120_DAYS           0\n",
       "RES_EL_CUR22_DAYS            0\n",
       "RES_EL_CUR30_DAYS            0\n",
       "RES_EL_CUR60_DAYS            0\n",
       "RES_EL_CUR90_DAYS            0\n",
       "RES_EL_CUR_BAL_AMT           0\n",
       "RES_EL_OVER_120_DAYS         0\n",
       "RES_GAS_CUR120_DAYS          0\n",
       "RES_GAS_CUR22_DAYS           0\n",
       "RES_GAS_CUR30_DAYS           0\n",
       "RES_GAS_CUR60_DAYS           0\n",
       "RES_GAS_CUR90_DAYS           0\n",
       "RES_GAS_CUR_BAL_AMT          0\n",
       "RES_GAS_OVER_120_DAYS        0\n",
       "BREAK_ARRANGEMENT            0\n",
       "BREAK_PAY_PLAN               0\n",
       "CALL_OUT                     0\n",
       "CALL_OUT_MANUAL              0\n",
       "DUE_DATE                     0\n",
       "FINAL_NOTICE                 0\n",
       "PAST_DUE                     0\n",
       "SEVERANCE_ELECTRIC           0\n",
       "SEVERANCE_GAS                0\n",
       "CITY_TOT_DUE                 0\n",
       "CITY_30_DAYS_PAST_DUE_AMT    0\n",
       "CITY_60_DAYS_PAST_DUE_AMT    0\n",
       "CITY_90_DAYS_PAST_DUE_AMT    0\n",
       "SPA_PER_ID                   0\n",
       "CMIS_MATCH                   0\n",
       "APARTMENT                    0\n",
       "ENROLL_DATE                  0\n",
       "HAS_COTENANT                 0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = df[~df.SPA_PER_ID.isna()]\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:29:59.058536Z",
     "start_time": "2021-07-16T21:29:59.044570Z"
    },
    "hidden": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "305"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df.CMIS_MATCH].SPA_PER_ID.nunique()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Check Identifiers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1    3242747\n",
       "2          2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.groupby(['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH']).size().value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>SPA_ACCT_ID</th>\n",
       "      <th>SPA_PREM_ID</th>\n",
       "      <th>MONTH</th>\n",
       "      <th>RES_EL_CUR120_DAYS</th>\n",
       "      <th>RES_EL_CUR22_DAYS</th>\n",
       "      <th>RES_EL_CUR30_DAYS</th>\n",
       "      <th>RES_EL_CUR60_DAYS</th>\n",
       "      <th>RES_EL_CUR90_DAYS</th>\n",
       "      <th>RES_EL_CUR_BAL_AMT</th>\n",
       "      <th>RES_EL_OVER_120_DAYS</th>\n",
       "      <th>...</th>\n",
       "      <th>SEVERANCE_GAS</th>\n",
       "      <th>CITY_TOT_DUE</th>\n",
       "      <th>CITY_30_DAYS_PAST_DUE_AMT</th>\n",
       "      <th>CITY_60_DAYS_PAST_DUE_AMT</th>\n",
       "      <th>CITY_90_DAYS_PAST_DUE_AMT</th>\n",
       "      <th>SPA_PER_ID</th>\n",
       "      <th>CMIS_MATCH</th>\n",
       "      <th>APARTMENT</th>\n",
       "      <th>ENROLL_DATE</th>\n",
       "      <th>HAS_COTENANT</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>111424</th>\n",
       "      <td>8062.0</td>\n",
       "      <td>86951.0</td>\n",
       "      <td>47</td>\n",
       "      <td>0.00</td>\n",
       "      <td>63.53</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>63.53</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>315.26</td>\n",
       "      <td>104.89</td>\n",
       "      <td>104.16</td>\n",
       "      <td>0.95</td>\n",
       "      <td>198608.0</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>()</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1110082</th>\n",
       "      <td>81799.0</td>\n",
       "      <td>42992.0</td>\n",
       "      <td>25</td>\n",
       "      <td>0.00</td>\n",
       "      <td>868.78</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>868.78</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>311.14</td>\n",
       "      <td>130.21</td>\n",
       "      <td>64.69</td>\n",
       "      <td>0.00</td>\n",
       "      <td>56253.0</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>()</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1915421</th>\n",
       "      <td>140997.0</td>\n",
       "      <td>86951.0</td>\n",
       "      <td>47</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>315.26</td>\n",
       "      <td>104.89</td>\n",
       "      <td>104.16</td>\n",
       "      <td>0.95</td>\n",
       "      <td>198608.0</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>()</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3611068</th>\n",
       "      <td>265974.0</td>\n",
       "      <td>42992.0</td>\n",
       "      <td>25</td>\n",
       "      <td>230.74</td>\n",
       "      <td>362.22</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>592.96</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>311.14</td>\n",
       "      <td>130.21</td>\n",
       "      <td>64.69</td>\n",
       "      <td>0.00</td>\n",
       "      <td>56253.0</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>()</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>4 rows × 35 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "         SPA_ACCT_ID  SPA_PREM_ID  MONTH  RES_EL_CUR120_DAYS  \\\n",
       "111424        8062.0      86951.0     47                0.00   \n",
       "1110082      81799.0      42992.0     25                0.00   \n",
       "1915421     140997.0      86951.0     47                0.00   \n",
       "3611068     265974.0      42992.0     25              230.74   \n",
       "\n",
       "         RES_EL_CUR22_DAYS  RES_EL_CUR30_DAYS  RES_EL_CUR60_DAYS  \\\n",
       "111424               63.53                0.0                0.0   \n",
       "1110082             868.78                0.0                0.0   \n",
       "1915421               0.00                0.0                0.0   \n",
       "3611068             362.22                0.0                0.0   \n",
       "\n",
       "         RES_EL_CUR90_DAYS  RES_EL_CUR_BAL_AMT  RES_EL_OVER_120_DAYS  ...  \\\n",
       "111424                 0.0               63.53                   0.0  ...   \n",
       "1110082                0.0              868.78                   0.0  ...   \n",
       "1915421                0.0                0.00                   0.0  ...   \n",
       "3611068                0.0              592.96                   0.0  ...   \n",
       "\n",
       "         SEVERANCE_GAS  CITY_TOT_DUE  CITY_30_DAYS_PAST_DUE_AMT  \\\n",
       "111424             0.0        315.26                     104.89   \n",
       "1110082            0.0        311.14                     130.21   \n",
       "1915421            0.0        315.26                     104.89   \n",
       "3611068            0.0        311.14                     130.21   \n",
       "\n",
       "         CITY_60_DAYS_PAST_DUE_AMT  CITY_90_DAYS_PAST_DUE_AMT  SPA_PER_ID  \\\n",
       "111424                      104.16                       0.95    198608.0   \n",
       "1110082                      64.69                       0.00     56253.0   \n",
       "1915421                     104.16                       0.95    198608.0   \n",
       "3611068                      64.69                       0.00     56253.0   \n",
       "\n",
       "         CMIS_MATCH  APARTMENT  ENROLL_DATE  HAS_COTENANT  \n",
       "111424        False      False           ()         False  \n",
       "1110082       False      False           ()          True  \n",
       "1915421       False      False           ()          True  \n",
       "3611068       False      False           ()         False  \n",
       "\n",
       "[4 rows x 35 columns]"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[df.duplicated(subset=['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH'], keep=False)]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Fix Duplicate Identifiers\n",
    "Problem  \n",
    "* Multiple instances of the same ('SPA_PER_ID', 'SPA_PREM_ID', 'MONTH') combination  \n",
    "\n",
    "Cause  \n",
    "* Different accounts at same time - person switched accounts for some reason  \n",
    "\n",
    "Solution  \n",
    "* Pick last"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original length: 3242751\n",
      "\n",
      "New length: 3242749\n",
      "\n",
      "2 Rows lost\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "1    3242749\n",
       "dtype: int64"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rows1 = len(df)\n",
    "print(f'Original length: {rows1}')\n",
    "print()\n",
    "df = df.groupby(['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH']).last().reset_index()\n",
    "print(f'New length: {len(df)}')\n",
    "print(f'\\n{rows1-len(df)} Rows lost')\n",
    "\n",
    "df.groupby(['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH']).size().value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "heading_collapsed": true
   },
   "source": [
    "# Geo - Avista"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:29:59.403801Z",
     "start_time": "2021-07-16T21:29:59.060527Z"
    },
    "hidden": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Total Records: 155538\n",
      "Total Premises: 155538\n",
      "Contains NaN's: False\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>SPA_PREM_ID</th>\n",
       "      <th>BLOCKGROUP_GEOID</th>\n",
       "      <th>POSTAL</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>24381.0</td>\n",
       "      <td>530630112013</td>\n",
       "      <td>99208</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>71746.0</td>\n",
       "      <td>530630024001</td>\n",
       "      <td>99201</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>148291.0</td>\n",
       "      <td>530630024001</td>\n",
       "      <td>99201</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>142249.0</td>\n",
       "      <td>530630105032</td>\n",
       "      <td>99208</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>33506.0</td>\n",
       "      <td>530630106024</td>\n",
       "      <td>99208</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   SPA_PREM_ID  BLOCKGROUP_GEOID  POSTAL\n",
       "0      24381.0      530630112013   99208\n",
       "1      71746.0      530630024001   99201\n",
       "2     148291.0      530630024001   99201\n",
       "3     142249.0      530630105032   99208\n",
       "4      33506.0      530630106024   99208"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "geo = pd.read_csv(datapath+'GeoData_Anon.csv').rename({'spa_prem_id':'SPA_PREM_ID'}, axis=1)\n",
    "\n",
    "geo = geo.drop([\"TRACT_GEOID\", \"BLOCKGROUP_GEOID_Data\"], axis=1).drop_duplicates()\n",
    "\n",
    "# NOTE: BLOCKGROUP_GEOID and BLOCKGROUP_GEOID_Data contain the same blockgroup number\n",
    "print(f'Total Records: {len(geo)}')\n",
    "print(f'Total Premises: {geo.SPA_PREM_ID.nunique()}')\n",
    "print(f\"Contains NaN's: {geo.isnull().any().any()}\")\n",
    "geo.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:30:01.223794Z",
     "start_time": "2021-07-16T21:29:59.406172Z"
    },
    "hidden": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "SPA_PER_ID                       0\n",
       "SPA_PREM_ID                      0\n",
       "MONTH                            0\n",
       "SPA_ACCT_ID                      0\n",
       "RES_EL_CUR120_DAYS               0\n",
       "RES_EL_CUR22_DAYS                0\n",
       "RES_EL_CUR30_DAYS                0\n",
       "RES_EL_CUR60_DAYS                0\n",
       "RES_EL_CUR90_DAYS                0\n",
       "RES_EL_CUR_BAL_AMT               0\n",
       "RES_EL_OVER_120_DAYS             0\n",
       "RES_GAS_CUR120_DAYS              0\n",
       "RES_GAS_CUR22_DAYS               0\n",
       "RES_GAS_CUR30_DAYS               0\n",
       "RES_GAS_CUR60_DAYS               0\n",
       "RES_GAS_CUR90_DAYS               0\n",
       "RES_GAS_CUR_BAL_AMT              0\n",
       "RES_GAS_OVER_120_DAYS            0\n",
       "BREAK_ARRANGEMENT                0\n",
       "BREAK_PAY_PLAN                   0\n",
       "CALL_OUT                         0\n",
       "CALL_OUT_MANUAL                  0\n",
       "DUE_DATE                         0\n",
       "FINAL_NOTICE                     0\n",
       "PAST_DUE                         0\n",
       "SEVERANCE_ELECTRIC               0\n",
       "SEVERANCE_GAS                    0\n",
       "CITY_TOT_DUE                     0\n",
       "CITY_30_DAYS_PAST_DUE_AMT        0\n",
       "CITY_60_DAYS_PAST_DUE_AMT        0\n",
       "CITY_90_DAYS_PAST_DUE_AMT        0\n",
       "CMIS_MATCH                       0\n",
       "APARTMENT                        0\n",
       "ENROLL_DATE                      0\n",
       "HAS_COTENANT                     0\n",
       "BLOCKGROUP_GEOID             11698\n",
       "POSTAL                       11698\n",
       "dtype: int64"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = df.join(geo.set_index('SPA_PREM_ID'), on=['SPA_PREM_ID'], how='left')\n",
    "del geo\n",
    "\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "hidden": true
   },
   "source": [
    "## Multi-Family Dwellings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:30:02.281479Z",
     "start_time": "2021-07-16T21:30:01.225720Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "dwellings = pd.read_csv(datapath+'MultiFamilyDwellingIDs_Anon.csv').rename({'spa_prem_id':'SPA_PREM_ID', 'multi_dwell_id':'MULTI_DWELL_ID'}, axis=1)\n",
    "df = df.join(dwellings.set_index('SPA_PREM_ID'), on=['SPA_PREM_ID'], how='left')\n",
    "del dwellings"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "heading_collapsed": true
   },
   "source": [
    "# Geo Data - Census\n",
    "Using data from 2015"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:30:02.296644Z",
     "start_time": "2021-07-16T21:30:02.284645Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "sub_datapath = datapath+'CensusData/'\n",
    "match_col = 'BLOCKGROUP_GEOID'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "hidden": true
   },
   "source": [
    "## Aggregate Income\n",
    "US Census Table: B19025"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:30:03.513506Z",
     "start_time": "2021-07-16T21:30:02.298607Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "agg_income = pd.read_csv(sub_datapath+'AggIncome/ACSDT5Y2015.B19025_data_with_overlays_2021-04-18T191340.csv')\n",
    "\n",
    "agg_income.drop(0, axis=0, inplace=True)\n",
    "newcol = \"AGG_INCOME_GEO\"\n",
    "agg_income.rename({\"B19025_001E\":newcol}, axis=1, inplace=True)\n",
    "\n",
    "agg_income[match_col] = agg_income[\"GEO_ID\"].map(preprocessing.geoid_map).astype('int64')\n",
    "agg_income.set_index(match_col, inplace=True)\n",
    "\n",
    "df = df.join(agg_income[newcol], how='left', on=match_col)\n",
    "\n",
    "del agg_income"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "hidden": true
   },
   "source": [
    "## Earnings\n",
    "US Census Table: B19051"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:30:04.616518Z",
     "start_time": "2021-07-16T21:30:03.513506Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "earnings = pd.read_csv(sub_datapath+'Earnings/ACSDT5Y2015.B19051_data_with_overlays_2021-04-12T234426.csv')\n",
    "\n",
    "earnings = earnings.drop(0, axis=0)\n",
    "newcol = \"NO_EARNINGS_GEO\"\n",
    "earnings[newcol] = earnings[\"B19051_003E\"].astype('float') / earnings[\"B19051_001E\"].astype('float')\n",
    "earnings[match_col] = earnings[\"GEO_ID\"].map(preprocessing.geoid_map).astype(\"int64\")\n",
    "earnings = earnings.set_index(match_col)\n",
    "\n",
    "df = df.join(earnings[newcol], how='left', on=match_col)\n",
    "del earnings"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "hidden": true
   },
   "source": [
    "## Poverty\n",
    "US Census Table B17021"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:30:05.750689Z",
     "start_time": "2021-07-16T21:30:04.618517Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "poverty = pd.read_csv(sub_datapath+'Poverty/ACSDT5Y2015.B17021_data_with_overlays_2021-04-12T234708.csv')\n",
    "\n",
    "poverty.drop(0, axis=0, inplace=True)\n",
    "newcol = \"BELOW_POVERTY_LVL_GEO\"\n",
    "poverty[newcol] = poverty[\"B17021_002E\"].astype('float') / poverty[\"B17021_001E\"].astype('float')\n",
    "poverty[match_col] = poverty[\"GEO_ID\"].map(preprocessing.geoid_map).astype('int64')\n",
    "poverty = poverty.set_index(match_col)\n",
    "\n",
    "df = df.join(poverty[newcol], how='left', on=match_col)\n",
    "del poverty"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "hidden": true
   },
   "source": [
    "## Food Stamps / SNAP\n",
    "US Census Table: B22010"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:30:06.886888Z",
     "start_time": "2021-07-16T21:30:05.752646Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "snap = pd.read_csv(sub_datapath+'FoodStamps/ACSDT5Y2015.B22010_data_with_overlays_2021-04-18T182516.csv')\n",
    "snap.drop(0, axis=0, inplace=True)\n",
    "newcol = \"SNAP_GEO\"\n",
    "snap[newcol] = snap[\"B22010_002E\"].astype('float') / snap[\"B22010_001E\"].astype('float')\n",
    "snap[match_col] = snap[\"GEO_ID\"].map(preprocessing.geoid_map).astype('int64')\n",
    "snap = snap.set_index(match_col)\n",
    "\n",
    "df= df.join(snap[newcol], how='left', on=match_col)\n",
    "del snap"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "hidden": true
   },
   "source": [
    "## Education Attainment\n",
    "US Census Table: B15003"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:30:08.056493Z",
     "start_time": "2021-07-16T21:30:06.890152Z"
    },
    "hidden": true,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "edu = pd.read_csv(sub_datapath+'Education/ACSDT5Y2015.B15003_data_with_overlays_2021-04-18T184101.csv')\n",
    "edu.columns = edu.iloc[0]\n",
    "edu = edu.drop(0, axis=0)\n",
    "# Get rid of margin of error columns\n",
    "for col in edu.columns:\n",
    "    if col == \"Margin of Error\":\n",
    "        edu = edu.drop(col, axis=1)\n",
    "\n",
    "newcol = \"ABOVE_GRD7_GEO\"\n",
    "# Sum of people above Grade 7 / total\n",
    "edu[newcol] = edu.iloc[:,13:].sum(axis=1) / edu.iloc[:,3:].sum(axis=1)\n",
    "\n",
    "edu[match_col] = edu[\"id\"].map(preprocessing.geoid_map).astype('int64')\n",
    "edu = edu.set_index(match_col)\n",
    "\n",
    "df = df.join(edu[newcol], how='left', on=match_col)\n",
    "del edu"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "hidden": true
   },
   "source": [
    "## Public Assistance¶\n",
    "US Census Table: B19057"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:30:09.198893Z",
     "start_time": "2021-07-16T21:30:08.059490Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "assist = pd.read_csv(sub_datapath+'PublicAssistance/ACSDT5Y2015.B19057_data_with_overlays_2021-04-18T190814.csv')\n",
    "assist.drop(0, axis=0, inplace=True)\n",
    "newcol = \"PUBLIC_ASSIST_GEO\"\n",
    "\n",
    "assist[newcol] = assist[\"B19057_002E\"].astype('float') / assist[\"B19057_001E\"].astype('float')\n",
    "assist[match_col] = assist[\"GEO_ID\"].map(preprocessing.geoid_map).astype('int64')\n",
    "assist = assist.set_index(match_col)\n",
    "\n",
    "df = df.join(assist[newcol], how='left', on=match_col)\n",
    "del assist"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "heading_collapsed": true
   },
   "source": [
    "# Check Data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "hidden": true
   },
   "source": [
    "## Check Nulls"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:30:10.051787Z",
     "start_time": "2021-07-16T21:30:09.204899Z"
    },
    "hidden": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "SPA_PER_ID                         0\n",
       "SPA_PREM_ID                        0\n",
       "MONTH                              0\n",
       "SPA_ACCT_ID                        0\n",
       "RES_EL_CUR120_DAYS                 0\n",
       "RES_EL_CUR22_DAYS                  0\n",
       "RES_EL_CUR30_DAYS                  0\n",
       "RES_EL_CUR60_DAYS                  0\n",
       "RES_EL_CUR90_DAYS                  0\n",
       "RES_EL_CUR_BAL_AMT                 0\n",
       "RES_EL_OVER_120_DAYS               0\n",
       "RES_GAS_CUR120_DAYS                0\n",
       "RES_GAS_CUR22_DAYS                 0\n",
       "RES_GAS_CUR30_DAYS                 0\n",
       "RES_GAS_CUR60_DAYS                 0\n",
       "RES_GAS_CUR90_DAYS                 0\n",
       "RES_GAS_CUR_BAL_AMT                0\n",
       "RES_GAS_OVER_120_DAYS              0\n",
       "BREAK_ARRANGEMENT                  0\n",
       "BREAK_PAY_PLAN                     0\n",
       "CALL_OUT                           0\n",
       "CALL_OUT_MANUAL                    0\n",
       "DUE_DATE                           0\n",
       "FINAL_NOTICE                       0\n",
       "PAST_DUE                           0\n",
       "SEVERANCE_ELECTRIC                 0\n",
       "SEVERANCE_GAS                      0\n",
       "CITY_TOT_DUE                       0\n",
       "CITY_30_DAYS_PAST_DUE_AMT          0\n",
       "CITY_60_DAYS_PAST_DUE_AMT          0\n",
       "CITY_90_DAYS_PAST_DUE_AMT          0\n",
       "CMIS_MATCH                         0\n",
       "APARTMENT                          0\n",
       "ENROLL_DATE                        0\n",
       "HAS_COTENANT                       0\n",
       "BLOCKGROUP_GEOID               11698\n",
       "POSTAL                         11698\n",
       "MULTI_DWELL_ID               2807756\n",
       "AGG_INCOME_GEO                 11698\n",
       "NO_EARNINGS_GEO                11698\n",
       "BELOW_POVERTY_LVL_GEO          11698\n",
       "SNAP_GEO                       11698\n",
       "ABOVE_GRD7_GEO                 11698\n",
       "PUBLIC_ASSIST_GEO              11698\n",
       "dtype: int64"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:30:12.361659Z",
     "start_time": "2021-07-16T21:30:10.055702Z"
    },
    "hidden": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "SPA_PER_ID                         0\n",
       "SPA_PREM_ID                        0\n",
       "MONTH                              0\n",
       "SPA_ACCT_ID                        0\n",
       "RES_EL_CUR120_DAYS                 0\n",
       "RES_EL_CUR22_DAYS                  0\n",
       "RES_EL_CUR30_DAYS                  0\n",
       "RES_EL_CUR60_DAYS                  0\n",
       "RES_EL_CUR90_DAYS                  0\n",
       "RES_EL_CUR_BAL_AMT                 0\n",
       "RES_EL_OVER_120_DAYS               0\n",
       "RES_GAS_CUR120_DAYS                0\n",
       "RES_GAS_CUR22_DAYS                 0\n",
       "RES_GAS_CUR30_DAYS                 0\n",
       "RES_GAS_CUR60_DAYS                 0\n",
       "RES_GAS_CUR90_DAYS                 0\n",
       "RES_GAS_CUR_BAL_AMT                0\n",
       "RES_GAS_OVER_120_DAYS              0\n",
       "BREAK_ARRANGEMENT                  0\n",
       "BREAK_PAY_PLAN                     0\n",
       "CALL_OUT                           0\n",
       "CALL_OUT_MANUAL                    0\n",
       "DUE_DATE                           0\n",
       "FINAL_NOTICE                       0\n",
       "PAST_DUE                           0\n",
       "SEVERANCE_ELECTRIC                 0\n",
       "SEVERANCE_GAS                      0\n",
       "CITY_TOT_DUE                       0\n",
       "CITY_30_DAYS_PAST_DUE_AMT          0\n",
       "CITY_60_DAYS_PAST_DUE_AMT          0\n",
       "CITY_90_DAYS_PAST_DUE_AMT          0\n",
       "CMIS_MATCH                         0\n",
       "APARTMENT                          0\n",
       "ENROLL_DATE                        0\n",
       "HAS_COTENANT                       0\n",
       "BLOCKGROUP_GEOID                   0\n",
       "POSTAL                             0\n",
       "MULTI_DWELL_ID               2797601\n",
       "AGG_INCOME_GEO                     0\n",
       "NO_EARNINGS_GEO                    0\n",
       "BELOW_POVERTY_LVL_GEO              0\n",
       "SNAP_GEO                           0\n",
       "ABOVE_GRD7_GEO                     0\n",
       "PUBLIC_ASSIST_GEO                  0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Drop all NA Geo info\n",
    "df = df[~df.POSTAL.isna()]\n",
    "\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "hidden": true
   },
   "source": [
    "## Check Data Types"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:30:12.377616Z",
     "start_time": "2021-07-16T21:30:12.364652Z"
    },
    "hidden": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "SPA_PER_ID                   float64\n",
       "SPA_PREM_ID                  float64\n",
       "MONTH                          int64\n",
       "SPA_ACCT_ID                  float64\n",
       "RES_EL_CUR120_DAYS           float64\n",
       "RES_EL_CUR22_DAYS            float64\n",
       "RES_EL_CUR30_DAYS            float64\n",
       "RES_EL_CUR60_DAYS            float64\n",
       "RES_EL_CUR90_DAYS            float64\n",
       "RES_EL_CUR_BAL_AMT           float64\n",
       "RES_EL_OVER_120_DAYS         float64\n",
       "RES_GAS_CUR120_DAYS          float64\n",
       "RES_GAS_CUR22_DAYS           float64\n",
       "RES_GAS_CUR30_DAYS           float64\n",
       "RES_GAS_CUR60_DAYS           float64\n",
       "RES_GAS_CUR90_DAYS           float64\n",
       "RES_GAS_CUR_BAL_AMT          float64\n",
       "RES_GAS_OVER_120_DAYS        float64\n",
       "BREAK_ARRANGEMENT            float64\n",
       "BREAK_PAY_PLAN               float64\n",
       "CALL_OUT                     float64\n",
       "CALL_OUT_MANUAL              float64\n",
       "DUE_DATE                     float64\n",
       "FINAL_NOTICE                 float64\n",
       "PAST_DUE                     float64\n",
       "SEVERANCE_ELECTRIC           float64\n",
       "SEVERANCE_GAS                float64\n",
       "CITY_TOT_DUE                 float64\n",
       "CITY_30_DAYS_PAST_DUE_AMT    float64\n",
       "CITY_60_DAYS_PAST_DUE_AMT    float64\n",
       "CITY_90_DAYS_PAST_DUE_AMT    float64\n",
       "CMIS_MATCH                      bool\n",
       "APARTMENT                       bool\n",
       "ENROLL_DATE                   object\n",
       "HAS_COTENANT                    bool\n",
       "BLOCKGROUP_GEOID             float64\n",
       "POSTAL                       float64\n",
       "MULTI_DWELL_ID               float64\n",
       "AGG_INCOME_GEO                object\n",
       "NO_EARNINGS_GEO              float64\n",
       "BELOW_POVERTY_LVL_GEO        float64\n",
       "SNAP_GEO                     float64\n",
       "ABOVE_GRD7_GEO               float64\n",
       "PUBLIC_ASSIST_GEO            float64\n",
       "dtype: object"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:30:17.231048Z",
     "start_time": "2021-07-16T21:30:12.380609Z"
    },
    "hidden": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "SPA_PER_ID                     int32\n",
       "SPA_PREM_ID                    int32\n",
       "MONTH                          int64\n",
       "SPA_ACCT_ID                    int32\n",
       "RES_EL_CUR120_DAYS           float64\n",
       "RES_EL_CUR22_DAYS            float64\n",
       "RES_EL_CUR30_DAYS            float64\n",
       "RES_EL_CUR60_DAYS            float64\n",
       "RES_EL_CUR90_DAYS            float64\n",
       "RES_EL_CUR_BAL_AMT           float64\n",
       "RES_EL_OVER_120_DAYS         float64\n",
       "RES_GAS_CUR120_DAYS          float64\n",
       "RES_GAS_CUR22_DAYS           float64\n",
       "RES_GAS_CUR30_DAYS           float64\n",
       "RES_GAS_CUR60_DAYS           float64\n",
       "RES_GAS_CUR90_DAYS           float64\n",
       "RES_GAS_CUR_BAL_AMT          float64\n",
       "RES_GAS_OVER_120_DAYS        float64\n",
       "BREAK_ARRANGEMENT              int32\n",
       "BREAK_PAY_PLAN                 int32\n",
       "CALL_OUT                       int32\n",
       "CALL_OUT_MANUAL                int32\n",
       "DUE_DATE                       int32\n",
       "FINAL_NOTICE                   int32\n",
       "PAST_DUE                       int32\n",
       "SEVERANCE_ELECTRIC             int32\n",
       "SEVERANCE_GAS                  int32\n",
       "CITY_TOT_DUE                 float64\n",
       "CITY_30_DAYS_PAST_DUE_AMT    float64\n",
       "CITY_60_DAYS_PAST_DUE_AMT    float64\n",
       "CITY_90_DAYS_PAST_DUE_AMT    float64\n",
       "CMIS_MATCH                      bool\n",
       "APARTMENT                       bool\n",
       "ENROLL_DATE                   object\n",
       "HAS_COTENANT                    bool\n",
       "BLOCKGROUP_GEOID               int32\n",
       "POSTAL                         int32\n",
       "MULTI_DWELL_ID                 int32\n",
       "AGG_INCOME_GEO               float64\n",
       "NO_EARNINGS_GEO              float64\n",
       "BELOW_POVERTY_LVL_GEO        float64\n",
       "SNAP_GEO                     float64\n",
       "ABOVE_GRD7_GEO               float64\n",
       "PUBLIC_ASSIST_GEO            float64\n",
       "dtype: object"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# If premesis not a multi-unit dwelling, set ID to -1\n",
    "df.MULTI_DWELL_ID = df.MULTI_DWELL_ID.replace(to_replace=np.nan, value=-1)\n",
    "\n",
    "to_ints = [\n",
    "    'BREAK_ARRANGEMENT',\n",
    "    'BREAK_PAY_PLAN',\n",
    "    'CALL_OUT',\n",
    "    'CALL_OUT_MANUAL',\n",
    "    'DUE_DATE',\n",
    "    'FINAL_NOTICE',\n",
    "    'PAST_DUE',\n",
    "    'SEVERANCE_ELECTRIC',\n",
    "    'SEVERANCE_GAS',\n",
    "    'SPA_PREM_ID',\n",
    "    'SPA_ACCT_ID',\n",
    "    'SPA_PER_ID',\n",
    "    'BLOCKGROUP_GEOID',\n",
    "    'POSTAL',\n",
    "    'MULTI_DWELL_ID',\n",
    "]\n",
    "for col in to_ints:\n",
    "    df[col] = df[col].astype('int')\n",
    "\n",
    "to_bools = [\n",
    "    'CMIS_MATCH',\n",
    "    'APARTMENT',\n",
    "    'HAS_COTENANT',\n",
    "]\n",
    "for col in to_bools:\n",
    "    df[col] = df[col].astype('bool')\n",
    "\n",
    "to_floats = [\n",
    "    'AGG_INCOME_GEO',\n",
    "]\n",
    "for col in to_floats:\n",
    "    df[col] = df[col].astype('float')\n",
    "\n",
    "df.dtypes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "heading_collapsed": true
   },
   "source": [
    "# Create Additional Attributes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## ID Combination"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['PER-PREM-MONTH_ID'] = df['SPA_PER_ID'].astype('str') + '-' + df['SPA_PREM_ID'].astype('str') + '-' + df['MONTH'].astype('str')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "hidden": true
   },
   "source": [
    "## Billing Aggregation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:30:17.435391Z",
     "start_time": "2021-07-16T21:30:17.232998Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "# Avista, City level\n",
    "df['AVISTA_CUR120_DAYS'] = df['RES_EL_CUR120_DAYS'] + df['RES_GAS_CUR120_DAYS']\n",
    "df['AVISTA_OVER_120_DAYS'] = df['RES_EL_OVER_120_DAYS'] + df['RES_GAS_OVER_120_DAYS']\n",
    "df['AVISTA_CUR22_DAYS'] = df['RES_EL_CUR22_DAYS'] + df['RES_GAS_CUR22_DAYS']\n",
    "df['AVISTA_CUR30_DAYS'] = df['RES_EL_CUR30_DAYS'] + df['RES_GAS_CUR30_DAYS']\n",
    "df['AVISTA_CUR60_DAYS'] = df['RES_EL_CUR60_DAYS'] + df['RES_GAS_CUR60_DAYS']\n",
    "df['AVISTA_CUR90_DAYS'] = df['RES_EL_CUR90_DAYS'] + df['RES_GAS_CUR90_DAYS']\n",
    "df['AVISTA_CUR_BAL_AMT'] = df['RES_EL_CUR_BAL_AMT'] + df['RES_GAS_CUR_BAL_AMT']\n",
    "\n",
    "# All\n",
    "df['TOTAL_30_DAYS_AMT'] = df['CITY_30_DAYS_PAST_DUE_AMT'] + df['AVISTA_CUR30_DAYS']\n",
    "df['TOTAL_60_DAYS_AMT'] = df['CITY_60_DAYS_PAST_DUE_AMT'] + df['AVISTA_CUR60_DAYS']\n",
    "df['TOTAL_90_DAYS_AMT'] = df['CITY_90_DAYS_PAST_DUE_AMT'] + df['AVISTA_CUR90_DAYS']\n",
    "df['TOTAL_CUR_BALANCE'] = df['AVISTA_CUR_BAL_AMT'] + df['CITY_TOT_DUE']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "hidden": true
   },
   "source": [
    "## Generate Different Outcome Measures\n",
    "### LAST_MO_W_DATA\n",
    "Last month with data on positive cases - estimate of when person started experiencing homelessness\n",
    "If have multiple ENROLL_DATEs, choose last month for each\n",
    "\n",
    "### 6_MO_PRIOR\n",
    "Within 6 months of last data month before experiencing homelessness?\n",
    "\n",
    "### MO_AWAY\n",
    "Number of months away from experiencing homelessness"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:30:17.451381Z",
     "start_time": "2021-07-16T21:30:17.438345Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "def get_outcomes(df):\n",
    "    '''\n",
    "    05/14/21\n",
    "    Creates\n",
    "        'LAST_MO_W_DATA' - boolean if P and last month with data\n",
    "        'WITHIN_6_MO_PRIOR_LAST_DATA' - boolean if P and within 6 mo of last data\n",
    "    '''\n",
    "    new_df = df.copy()\n",
    "    lasts = new_df[new_df.CMIS_MATCH].groupby(['SPA_PER_ID']).last()[['SPA_PREM_ID', 'MONTH']]\n",
    "    lasts['LAST_MO_W_DATA'] = lasts['MONTH']\n",
    "    lasts = lasts.reset_index().set_index(['SPA_PER_ID', 'SPA_PREM_ID']).drop('MONTH', axis=1)\n",
    "    new_df = new_df.join(lasts, on=['SPA_PER_ID', 'SPA_PREM_ID'], how='left')\n",
    "    # Create WITHIN_6_MO_PRIOR_LAST_DATA\n",
    "    new_df['WITHIN_6_MO_PRIOR_LAST_DATA'] = (new_df['MONTH'] >= (new_df['LAST_MO_W_DATA'] - 6))\n",
    "    # Create MO_AWAY\n",
    "    new_df['MO_AWAY'] = new_df['LAST_MO_W_DATA'] - new_df['MONTH']\n",
    "    # Change 'LAST_MO_W_DATA' to boolean\n",
    "    new_df['LAST_MO_W_DATA'] = (new_df['LAST_MO_W_DATA'] == new_df['MONTH']).replace(to_replace=np.nan, value=False).astype('bool')\n",
    "    return new_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:30:19.727358Z",
     "start_time": "2021-07-16T21:30:17.454395Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "df = get_outcomes(df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "hidden": true
   },
   "source": [
    "## People and Premises\n",
    "* 'NUM_SPA_PER_ID_FOR_SPA_PREM_ID': number of people for each premises\n",
    "* 'NUM_SPA_PREM_ID_FOR_SPA_PER_ID': number of premises for each person"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:30:19.742648Z",
     "start_time": "2021-07-16T21:30:19.729708Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "def accumulate(df, grp_by_col, cumulative_col, new_col_name):\n",
    "    '''\n",
    "    05/02/21\n",
    "    Finds cumulative counts.\n",
    "    '''\n",
    "    month_col = 'MONTH'\n",
    "    cumulative = df[[month_col, grp_by_col, cumulative_col]].copy()\n",
    "    # Find number of unique cumulateive elements\n",
    "    cumulative = cumulative.drop_duplicates([grp_by_col, cumulative_col], keep='first').groupby([grp_by_col, month_col]).nunique()\n",
    "    # Find cumulative count of unique elements\n",
    "    cumulative[new_col_name] = (cumulative.groupby(grp_by_col)[cumulative_col].cumcount() + 1).astype('int64')\n",
    "    cumulative.drop(cumulative_col, axis=1, inplace=True)\n",
    "    # Join counts back to df\n",
    "    new_df = df.join(cumulative, how='left', on=[grp_by_col, month_col])\n",
    "    # Forward fill index gaps\n",
    "    new_df[new_col_name].ffill(inplace=True)\n",
    "    return new_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:30:24.759957Z",
     "start_time": "2021-07-16T21:30:19.744647Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "# Determine cumulative number of places a person has paid bills at so far\n",
    "df = accumulate(df, grp_by_col='SPA_PER_ID', cumulative_col='SPA_PREM_ID', new_col_name='NUM_PREM_FOR_PER')\n",
    "\n",
    "# Determine cumulative number of people a premesis has seen so far\n",
    "df = accumulate(df, grp_by_col='SPA_PREM_ID', cumulative_col='SPA_PER_ID', new_col_name='NUM_PER_FOR_PREM')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "hidden": true
   },
   "source": [
    "## Size of MultiUnit¶\n",
    "number of SPA_PREM_ID's at same MULTI_DWELL_ID"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:30:26.100919Z",
     "start_time": "2021-07-16T21:30:24.762125Z"
    },
    "hidden": true
   },
   "outputs": [],
   "source": [
    "multi_dwell_size = df.groupby('MULTI_DWELL_ID').SPA_PREM_ID.nunique().rename('MULTI_DWELL_SIZE')\n",
    "# Set size for not multi_unit to 0 \n",
    "multi_dwell_size.loc[np.nan] = 0\n",
    "df = df.join(multi_dwell_size, how='left', on='MULTI_DWELL_ID')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Get Preprocessing Stats and Save"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Print Columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['SPA_PER_ID',\n",
       " 'SPA_PREM_ID',\n",
       " 'MONTH',\n",
       " 'SPA_ACCT_ID',\n",
       " 'RES_EL_CUR120_DAYS',\n",
       " 'RES_EL_CUR22_DAYS',\n",
       " 'RES_EL_CUR30_DAYS',\n",
       " 'RES_EL_CUR60_DAYS',\n",
       " 'RES_EL_CUR90_DAYS',\n",
       " 'RES_EL_CUR_BAL_AMT',\n",
       " 'RES_EL_OVER_120_DAYS',\n",
       " 'RES_GAS_CUR120_DAYS',\n",
       " 'RES_GAS_CUR22_DAYS',\n",
       " 'RES_GAS_CUR30_DAYS',\n",
       " 'RES_GAS_CUR60_DAYS',\n",
       " 'RES_GAS_CUR90_DAYS',\n",
       " 'RES_GAS_CUR_BAL_AMT',\n",
       " 'RES_GAS_OVER_120_DAYS',\n",
       " 'BREAK_ARRANGEMENT',\n",
       " 'BREAK_PAY_PLAN',\n",
       " 'CALL_OUT',\n",
       " 'CALL_OUT_MANUAL',\n",
       " 'DUE_DATE',\n",
       " 'FINAL_NOTICE',\n",
       " 'PAST_DUE',\n",
       " 'SEVERANCE_ELECTRIC',\n",
       " 'SEVERANCE_GAS',\n",
       " 'CITY_TOT_DUE',\n",
       " 'CITY_30_DAYS_PAST_DUE_AMT',\n",
       " 'CITY_60_DAYS_PAST_DUE_AMT',\n",
       " 'CITY_90_DAYS_PAST_DUE_AMT',\n",
       " 'CMIS_MATCH',\n",
       " 'APARTMENT',\n",
       " 'ENROLL_DATE',\n",
       " 'HAS_COTENANT',\n",
       " 'BLOCKGROUP_GEOID',\n",
       " 'POSTAL',\n",
       " 'MULTI_DWELL_ID',\n",
       " 'AGG_INCOME_GEO',\n",
       " 'NO_EARNINGS_GEO',\n",
       " 'BELOW_POVERTY_LVL_GEO',\n",
       " 'SNAP_GEO',\n",
       " 'ABOVE_GRD7_GEO',\n",
       " 'PUBLIC_ASSIST_GEO',\n",
       " 'PER-PREM-MONTH_ID',\n",
       " 'AVISTA_CUR120_DAYS',\n",
       " 'AVISTA_OVER_120_DAYS',\n",
       " 'AVISTA_CUR22_DAYS',\n",
       " 'AVISTA_CUR30_DAYS',\n",
       " 'AVISTA_CUR60_DAYS',\n",
       " 'AVISTA_CUR90_DAYS',\n",
       " 'AVISTA_CUR_BAL_AMT',\n",
       " 'TOTAL_30_DAYS_AMT',\n",
       " 'TOTAL_60_DAYS_AMT',\n",
       " 'TOTAL_90_DAYS_AMT',\n",
       " 'TOTAL_CUR_BALANCE',\n",
       " 'LAST_MO_W_DATA',\n",
       " 'WITHIN_6_MO_PRIOR_LAST_DATA',\n",
       " 'MO_AWAY',\n",
       " 'NUM_PREM_FOR_PER',\n",
       " 'NUM_PER_FOR_PREM',\n",
       " 'MULTI_DWELL_SIZE']"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns.to_list()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Check Nulls"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.drop('MO_AWAY', axis=1).isnull().sum().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Save Pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:30:38.338952Z",
     "start_time": "2021-07-16T21:30:26.102913Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\"\\nfilename = 'processed.pickle'\\noutfile = open(datapath+filename, 'wb')\\npickle.dump(df, outfile)\\noutfile.close()\\n\""
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''\n",
    "filename = 'processed.pickle'\n",
    "outfile = open(datapath+filename, 'wb')\n",
    "pickle.dump(df, outfile)\n",
    "outfile.close()\n",
    "'''"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Check Numbers Retained"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:30:39.255999Z",
     "start_time": "2021-07-16T21:30:38.340923Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Retained 3231051 = 84.55853603769822% of rows.\n",
      "Retained 85249 = 86.94086931690701% of accounts.\n",
      "Retained 302 = 12.651864264767491% of positive cases.\n",
      "People: 84345\n",
      "Negative Cases: 84066\n"
     ]
    }
   ],
   "source": [
    "retained_rows = len(df)\n",
    "retained_accts = df.SPA_ACCT_ID.nunique()\n",
    "retained_pos_cases = df[df.CMIS_MATCH].SPA_PER_ID.nunique()\n",
    "\n",
    "print(f'Retained {retained_rows} = {100*retained_rows/rows}% of rows.')\n",
    "print(f'Retained {retained_accts} = {100*retained_accts/accts}% of accounts.')\n",
    "print(f'Retained {retained_pos_cases} = {100*retained_pos_cases/sa_pos_ppl}% of positive cases.')\n",
    "print(f'People: {df.SPA_PER_ID.nunique()}')\n",
    "print(f'Negative Cases: {df[~df.CMIS_MATCH].SPA_PER_ID.nunique()}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.1613338230062089"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "84345 / 522798"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Total Time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-07-16T21:30:39.271613Z",
     "start_time": "2021-07-16T21:30:39.258648Z"
    }
   },
   "outputs": [],
   "source": [
    "calc_time.calc_time_from_sec(time.time()-startTime)"
   ]
  }
 ],
 "metadata": {
  "interpreter": {
   "hash": "55dc2ef33d481be31a5c00b42f248808e4edd8b163c72100e3df152a03ba29f1"
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.12"
  },
  "metadata": {
   "interpreter": {
    "hash": "55dc2ef33d481be31a5c00b42f248808e4edd8b163c72100e3df152a03ba29f1"
   }
  },
  "varInspector": {
   "cols": {
    "lenName": 16,
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
