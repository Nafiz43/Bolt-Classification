import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from numpy import inf


app = Flask(__name__,static_url_path = "/tmp", static_folder = "tmp")

dt_model = pickle.load(open('bolt_dt_c.pkl', 'rb'))
rf_model = pickle.load(open('bolt_rf_c.pkl', 'rb'))
gnb_model = pickle.load(open('bolt_gnb_c.pkl','rb'))
svm_model = pickle.load(open('bolt_svm_c.pkl','rb'))
knn_model = pickle.load(open('bolt_knn_c.pkl','rb'))
ann_model = pickle.load(open('bolt_ann_c.pkl','rb'))
ab_model = pickle.load(open('bolt_ab_c.pkl','rb'))
cb_model = pickle.load(open('bolt_cb_c.pkl','rb'))
gb_model = pickle.load(open('bolt_gb_c.pkl','rb'))
xb_model = pickle.load(open('bolt_xb_c.pkl','rb'))



scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')




@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    int_features = [float(x) for x in request.form.values()]
    print(int_features)
    
    #print(int_features)
    e1_do=int_features[0]/int_features[2]
    e2_do=int_features[1]/int_features[2]
    fu_fy = int_features[3]/int_features[4]
    p1_do = int_features[5]/int_features[2]
    p2_do = int_features[6]/int_features[2]
    nr = int_features[7]
    #print(e1_do,e2_do,fu_fy,type_c)
    #print(int_features[5])
    final_features=[]
    
    #final_features=final_features+[np.log(e2_do)]
    #final_features=final_features+[np.log(fu_fy)]
    
    #e1/do	e2/do	p1/do	p2/do	Nr	fu/fy
    
    
    final_features=final_features+[(e1_do)]         
    final_features=final_features+[(e2_do)]
    final_features=final_features+[p1_do]
    final_features=final_features+[p2_do]
    final_features=final_features+[nr]
    final_features=final_features+[(fu_fy)]
    

        
 
    
    
    
    final_features = [np.array(final_features)]
    print(final_features)
    

    #final_features=np.log(final_features)

    #print(final_features)
    final_features=scaler.transform(final_features)
    
    
    print(final_features)
    
    
    dt_prediction = dt_model.predict(final_features)
    rf_prediction = rf_model.predict(final_features)
    gnb_prediction= gnb_model.predict(final_features)
    svm_prediction = svm_model.predict(final_features)
    knn_prediction=  knn_model.predict(final_features)
    ann_prediction= ann_model.predict(final_features)
    ab_prediction = ab_model.predict(final_features)
    cb_prediction = cb_model.predict(final_features)
    gb_prediction = gb_model.predict(final_features)
    xb_prediction = xb_model.predict(final_features)
        
    
    dt_prediction = dt_prediction[0]
    rf_prediction = rf_prediction[0]
    gnb_prediction= gnb_prediction[0]
    svm_prediction = svm_prediction[0]
    knn_prediction= knn_prediction[0]
    ann_prediction= ann_prediction[0]
    ab_prediction = ab_prediction[0]
    cb_prediction = cb_prediction[0]
    cb_prediction = cb_prediction[0]
    gb_prediction = gb_prediction[0]
    xb_prediction = xb_prediction[0]
    
   
    if (dt_prediction==0):
        dt_prediction='B'
    elif (dt_prediction==1):
        dt_prediction='N'
    elif (dt_prediction==2):
        dt_prediction='SP'
    elif (dt_prediction==3):
        dt_prediction='TO'
        
    if (rf_prediction==0):
        rf_prediction='B'
    elif (rf_prediction==1):
        rf_prediction='N'
    elif (rf_prediction==2):
        rf_prediction='SP'
    elif (rf_prediction==3):
        rf_prediction='TO'
        
    if (gnb_prediction==0):
        gnb_prediction='B'
    elif (gnb_prediction==1):
        gnb_prediction='N'
    elif (gnb_prediction==2):
        gnb_prediction='SP'
    elif (gnb_prediction==3):
        gnb_prediction='TO'   

    if (svm_prediction==0):
        svm_prediction='B'
    elif (svm_prediction==1):
        svm_prediction='N'
    elif (svm_prediction==2):
        svm_prediction='SP'
    elif (svm_prediction==3):
        svm_prediction='TO'
        
        
    
    if (knn_prediction==0):
        knn_prediction='B'
    elif (knn_prediction==1):
        knn_prediction='N'
    elif (knn_prediction==2):
        knn_prediction='SP'
    elif (knn_prediction==3):
        knn_prediction='TO'
    
    if (ann_prediction==0):
        ann_prediction='B'
    elif (ann_prediction==1):
        ann_prediction='N'
    elif (ann_prediction==2):
        ann_prediction='SP'
    elif (ann_prediction==3):
        ann_prediction='TO'
        
        
    if (ab_prediction==0):
        ab_prediction='B'
    elif (ab_prediction==1):
        ab_prediction='N'
    elif (ab_prediction==2):
        ab_prediction='SP'
    elif (ab_prediction==3):
        ab_prediction='TO'
        
        
        
    if (cb_prediction==0):
        cb_prediction='B'
    elif (cb_prediction==1):
        cb_prediction='N'
    elif (cb_prediction==2):
        cb_prediction='SP'
    elif (cb_prediction==3):
        cb_prediction='TO'
    
    
    if (gb_prediction==0):
        gb_prediction='B'
    elif (gb_prediction==1):
        gb_prediction='N'
    elif (gb_prediction==2):
        gb_prediction='SP'
    elif (gb_prediction==3):
        gb_prediction='TO'
    
    
    if (xb_prediction==0):
        xb_prediction='B'
    elif (xb_prediction==1):
        xb_prediction='N'
    elif (xb_prediction==2):
        xb_prediction='SP'
    elif (xb_prediction==3):
        xb_prediction='TO'
    
    
        

    
    '''
    ab_prediction_o = round(ab_prediction[0], 3)
    ann_prediction_o = round(ann_prediction[0], 3)
    cb_prediction_o = round(cb_prediction[0], 3)
    dt_prediction_o = round(dt_prediction[0], 3)
    knn_prediction_o = round(knn_prediction[0], 3)
    lasso_prediction_o=round(lasso_prediction[0], 3)
    lr_prediction_o = round(lr_prediction[0], 3)
    ridge_prediction_o = round(ridge_prediction[0], 3)
    svr_prediction_o = round(svr_prediction[0], 3)
    xg_prediction_o = round(xg_prediction[0], 3)
    
    rf_prediction_o=format(rf_prediction_o,'.3f')
    ab_prediction_o=format(ab_prediction_o,'.3f')
    ann_prediction_o=format(ann_prediction_o, '.3f')
    cb_prediction_o=format(cb_prediction_o, '.3f')
    dt_prediction_o=format(dt_prediction_o, '.3f')
    knn_prediction_o=format(knn_prediction_o, '.3f')
    lasso_prediction_o=format(lasso_prediction_o, '.3f')
    lr_prediction_o=format(lr_prediction_o, '.3f')
    ridge_prediction_o=format(ridge_prediction_o, '.3f')
    svr_prediction_o=format(svr_prediction_o, '.3f')
    xg_prediction_o=format(xg_prediction_o, '.3f')
    
    '''
    print(dt_prediction)
    print(dt_prediction, rf_prediction, gnb_prediction, svm_prediction, knn_prediction, ann_prediction, ab_prediction, cb_prediction, gb_prediction, xb_prediction)
    #print(dt_prediction,ab_prediction_o,ann_prediction_o,cb_prediction_o,dt_prediction_o,knn_prediction_o,lasso_prediction_o,lr_prediction_o,ridge_prediction_o,svr_prediction_o,xg_prediction_o)
    

    
    return render_template('index.html', dt='{}'.format(dt_prediction), rf='{}'.format(rf_prediction), gnb='{}'.format(gnb_prediction), svm='{}'.format(svm_prediction), knn='{}'.format(knn_prediction), ann='{}'.format(ann_prediction), ab='{}'.format(ab_prediction), cb='{}'.format(cb_prediction), gb='{}'.format(gb_prediction), xb='{}'.format(xb_prediction)) 


if __name__ == "__main__":
    app.run(debug=True)