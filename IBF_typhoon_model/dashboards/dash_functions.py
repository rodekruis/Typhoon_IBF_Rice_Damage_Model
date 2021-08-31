#%%Importing libraries
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from cvxopt import matrix, solvers
import cvxopt
import dash_html_components as html


#%%Dash table
def make_dash_table(df):
    """ Return a dash definition of an HTML table for a Pandas dataframe """
    table = []
    for index, row in df.iterrows():
        html_row = []
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table

#%%Gaussian kernel function
def gaussian_kernel(x,y,gamma):
    return np.exp(-(np.linalg.norm((x.T-y.T))**2)*gamma)

#%%Manual SVR calculation using cvxopt
def epsilon_svr(X_train,Y_train,train_prob,X_test,C,gamma,epsilon):
    
    N = X_train.shape[0]
    C = float(C)

    #Finding the kernels i.e. k(x,x')
    k = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            k[i][j] = gaussian_kernel(X_train[i,:], X_train[j,:], gamma)
                
    #Making matrices P,q,A, G, h
    element1 = k
    P= np.concatenate((element1,-1*element1),axis=1)
    P= np.concatenate((P,-1*P),axis=0)
    
    q= epsilon*np.ones((2*N,1))
    qadd=np.concatenate((-1*Y_train,Y_train),axis=0)
    q=q+qadd
    
    A=np.concatenate((np.ones((1,N)),-1*(np.ones((1,N)))),axis=1)
    
    G=np.concatenate((np.eye(2*N),-1*np.eye(2*N)),axis=0)

    # h=np.concatenate((C*np.ones((2*N,1)),np.zeros((2*N,1))),axis=0)
    h = np.concatenate((C*train_prob,np.zeros((2*N,1))),axis=0)  

    print(f"P: {P.shape}")
    print(f"q: {q.shape}")
    print(f"G: {G.shape}")
    print(f"h: {h.shape}")
    print(f"A: {A.shape}")

    #define matrices for optimization problem       
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    A = cvxopt.matrix(A)
    G=cvxopt.matrix(G)
    h = cvxopt.matrix(h)
    b = cvxopt.matrix(np.zeros((1,1)))
    
    #solve the optimization problem
    sol = cvxopt.solvers.qp(P,q,G,h,A,b,solver='glpk')
    
    #getting the lagrange multipliers
    l = np.ravel(sol['x'])
    #parting to get the 2 sets of Lagrange multipliers
    u=l[0:N]
    v=l[N:]    
    #Getting the  support vectors
    u1=u > 1e-5
    v1=v > 1e-5
    SV=np.logical_or(u1, v1)
    SVindices = np.arange(len(l)/2)[SV] 
    u1=u[SVindices.astype(int)]
    v1=v[SVindices.astype(int)]
    support_vectors_x = X_train[SV]
    support_vectors_y = Y_train[SV]    
    #calculate intercept
    bias= sol['y']
    
    #find weight vector and predict y
    Y_pred = np.zeros((len(X_test),1))
    for i in range(len(X_test)):
        val=0
        for u_,v_,z in zip(u1,v1,support_vectors_x):
            val=val+(u_ - v_)*gaussian_kernel(X_test[i],z,gamma)
        Y_pred[i,0]= val
    Y_pred = Y_pred+bias[0,0]
    
    return Y_pred   

#%%SVR function
def svr_dash(df_original, typhoons, mun_codes, features_used, gamma, C, epsilon):

    df = df_original
    
    df_results = pd.DataFrame({'mun_codes':mun_codes})
    test_score = []

    for typhoon in typhoons:
        print(typhoon)

        train = df[df['typhoons']!=typhoon]
        test = df[df['typhoons']==typhoon]
        y_train = train['perc_loss']
        x_train = train[features_used]
        y_test = test['perc_loss']
        x_test = test[features_used]
        #Scale the x variable
        x_scaler = StandardScaler()
        x_train_scaled = x_scaler.fit_transform(x_train)
        x_test_scaled = x_scaler.transform(x_test)
        #Scale the y variable
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(np.asarray(y_train).reshape(-1,1)).ravel()

        trained_svr = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon).fit(x_train_scaled, y_train_scaled)
        
        y_pred_test =  trained_svr.predict(x_test_scaled)
        y_pred_test = y_scaler.inverse_transform(y_pred_test)
        df_results[typhoon] = y_pred_test
       
        test_score.append(mean_absolute_error(y_pred_test, y_test))

    test_score.append(np.mean(test_score))

    return df_results, test_score

#RWSVR function
def rwsvr_dash(df_original, df_prob, typhoons, mun_codes, features_used, C, gamma, epsilon):
    
    df = df_original
    df_results = pd.DataFrame({'mun_codes':mun_codes})
    test_score = []

    for typhoon in typhoons:
        print(typhoon)    
        train = df[df['typhoons']!=typhoon]
        test = df[df['typhoons']==typhoon]
        y_train = train['perc_loss']
        x_train = train[features_used]
        y_test = test['perc_loss']
        x_test = test[features_used]

        scaler_x = StandardScaler()
        x_train_scaled = scaler_x.fit_transform(x_train)
        x_test_scaled = scaler_x.transform(x_test)
        
        scaler_y = StandardScaler()
        y_train = np.asarray(y_train).reshape(-1, 1)
        y_test = np.asarray(y_test).reshape(-1, 1)
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)

        X_train = np.asarray(x_train_scaled)
        X_test = np.asarray(x_test_scaled)
        Y_train = np.asarray(y_train_scaled).reshape(X_train.shape[0],1)
        Y_train_new = Y_train.ravel()

        df_prob_used = df_prob.drop(columns=[typhoon, 'mun_codes', 'Unnamed: 0'])
        df_prob_used = df_prob_used.stack().reset_index()

        train_prob = np.asarray(df_prob_used[0]).reshape(X_train.shape[0], 1) ** ()
        train_prob_double = np.concatenate((train_prob, train_prob), axis=0)

        result = epsilon_svr(X_train,Y_train,train_prob_double,X_test,C=C,gamma=gamma,epsilon=epsilon)
        y_pred_test = scaler_y.inverse_transform(result)

        df_results[typhoon] = y_pred_test
        print(mean_absolute_error(y_pred_test, y_test))
        test_score.append(mean_absolute_error(y_pred_test, y_test))

    return df_results, test_score




#%%
df = pd.read_excel("C:\\Users\\Marieke\\GitHub\\Rice_Field_Damage_Philippines\\app\\data\\data_output.xlsx")

df_proba = pd.read_excel("C:\\Users\\Marieke\\GitHub\\Rice_Field_Damage_Philippines\\app\\outputs\\svm\\predicted_probs.xlsx")

features_used = ['dist_track', 'vmax_sust', 'mean_ruggedness', 'ruggedness_stdev']

mun_codes = df['municipality_codes'].unique()
typhoons = df['typhoons'].unique().tolist()

df_results, test_score =rwsvr_dash(df, df_proba, typhoons, mun_codes, features_used, C=4, gamma=0.5, epsilon=0.01)

#%%
df_results_svr, test_score = svr_dash(df, typhoons, mun_codes, features_used, gamma=0.5, C=4, epsilon=0.5)


#%%
def weighted_mae(y_pred, y_actual):
    
    df = pd.DataFrame({'actual': y_actual, 'predicted':y_pred})
    total = sum(y_actual)
    df['error'] = np.abs(y_pred - y_actual)
    df['weight'] = y_actual / total
    df['weighted_error'] = df['error'] * df['weight']

    weighted_mae = sum(df['weighted_error'])

    return weighted_mae


wmae_rwsvr = []
wmae_svr = []

for typhoon in typhoons:
    test = df[df['typhoons']==typhoon]
    y_test = test['perc_loss'].reset_index(drop=True)
    y_pred = df_results[typhoon]

    wmae = weighted_mae(y_pred, y_test)
    wmae_rwsvr.append(wmae)


# %%
for typhoon in typhoons:
    test = df[df['typhoons']==typhoon]
    y_test = test['perc_loss'].reset_index(drop=True)
    y_pred = df_results_svr[typhoon]

    wmae = weighted_mae(y_pred, y_test)
    wmae_svr.append(wmae)

# %%
