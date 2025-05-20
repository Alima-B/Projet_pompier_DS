# test de faire comme dans le nb

import streamlit as st
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from feature_engine.encoding import MeanEncoder
from sklearn.preprocessing import RobustScaler

from sklearn.metrics import mean_squared_error,mean_absolute_error, root_mean_squared_error,r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from imblearn.metrics import macro_averaged_mean_absolute_error ,classification_report_imbalanced, geometric_mean_score , sensitivity_score

import shap
import pickle



# Fonction pour lire l'image et la convertir en base64
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Utiliser l'image de fond
background_image = "Images/pompiers_londres.jpg"
bg_image_base64 = get_base64_of_bin_file(background_image)

# Définir le style CSS pour la page avec une superposition
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{bg_image_base64}");
    background-size: cover;
    background-position: center;
    height: 100vh;
    position: relative;
}}
[data-testid="stHeader"] {{
    background: rgba(0, 0, 0, 0);
}}
.overlay {{
    position: fixed; /* Changer de absolute à fixed */
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.4); /* Couleur noire avec 40% de transparence */
    z-index: 1;
}}
.container {{
    position: relative;
    z-index: 2;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
}}
.title {{
    color: white;
    font-size: 3em;
    text-align: center;
    margin: 0;
}}
</style>
"""

# fonction pour charger le df Mobilisation et le df Mobilisation

pages = ["Home","Introduction","Modélisation"] 
st.sidebar.title("Sommaire")
page = st.sidebar.radio("Allez vers", pages)


# Titre de la page
if page == pages[0] :


    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown('<div class="overlay"></div>', unsafe_allow_html=True)
    st.markdown('<div class="container"><h1 class="title">Projet Pompiers de la ville de Londres</h1></div>', unsafe_allow_html=True)
 

# Page d'introduction
if page == pages[1] :

    st.title("Introduction")
    st.markdown(
        """
        **Bienvenue dans le projet "Pompiers de la ville de Londres".**

        La Brigade des Pompiers de Londres, avec 5000 personnes, est le plus grand service
        d'incendie et de secours du pays, protégeant le Grand Londres. Suite aux incendies
        meurtriers passés, des investissements croissants ont été réalisés, le temps de réaction
        étant crucial. Les incendies, souvent débutant par une petite flamme contrôlable,
        se propagent avec le temps, aggravant les conditions de lutte. Le facteur essentiel est
        le temps de réponse des pompiers, crucial pour la survie et la minimisation des dégâts.

        Ces dernières années, l'intégration de l'IA dans les services d'incendie s'avère prometteuse.
        L'IA permet la prévision des incendies et la prise de décisions en temps réel grâce à l'analyse
        des données. Les algorithmes identifient les risques, prédisent la propagation, aidant
        les pompiers à élaborer des stratégies proactives et à allouer les ressources de manière optimale.
        L'IA fournit également des données en temps réel améliorant la connaissance de la situation.
        """
    )
    
    st.subheader("Objectif du Projet")
    st.markdown(
        """
        Ce projet vise à analyser et à visualiser les données relatives aux interventions des pompiers
        dans la ville de Londres. Nous nous concentrons sur les types d'interventions, les tendances au
        fil du temps, et les facteurs influençant les demandes d'intervention. Suite à cette analyse, l'objectif
        a été centré sur la prédiction du temps de réponse de la première brigade arrivant sur un incident à la suite d’un appel.   
        
        Cette application présente les performances de modèles de régression et de classification et permet de faire des prédictions.
        """
    )

     

def rapport_linreg(model, X_test, y_test, order,dico):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, labels=range(len(order)), target_names=order, output_dict=True)
    y_pred_adj = [dico[i] for i in y_pred]
    mat = pd.crosstab(y_test.replace(dico), y_pred_adj, rownames=['Classe réelle'], colnames=['Classe prédite']).reindex(index=order ,columns=order)
    rapport_df = pd.DataFrame(class_report).transpose()
    return accuracy, mat, rapport_df

def rapport_randomforest(model, X_test, y_test, order,dico):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, labels=range(len(order)), target_names=order, output_dict=True)
    y_pred_adj = [dico[i] for i in y_pred]
    mat = pd.crosstab(y_test.replace(dico), y_pred_adj, rownames=['Classe réelle'], colnames=['Classe prédite']).reindex(index=order ,columns=order)
    rapport_df = pd.DataFrame(class_report).transpose()
    return accuracy, mat, rapport_df


def rapport_mat2(model_name, X_test, y_test, dico_model, order):
    model = dico_model.get(model_name)
    if model is None:
        st.error(f"Modèle {model_name} non trouvé.")
        return None, None, None, None

    elif model_name == "RandomForestClassifier":
        return rapport_randomforest(model, X_test, y_test, order,dico)
    

# Page de modélisation
if page == pages[2] :
    
    # le dataFrame avec les données encodées et filtre 2020
    @st.cache_data
    def Data_Mod(chemin):
        
        dico = {'IncidentNumber' : 'str',     
            }
        df = pd.read_csv(chemin, dtype = dico)
        # ncr de rename les colonnes avec les noms d'origine car les nouvelles étaient utilisées pour conserver les  valeurs originales après un pd.getdummies
        df = df.rename(columns={"IncidentGroup_orig" : "IncidentGroup",
                        "StopCodeDescription_orig" : "StopCodeDescription",
                        "PropertyCategory_orig" : "PropertyCategory"})

        df["dst_StationIncident"] = df["dst_StationIncident"]/1000

        return df

    df_2020 = Data_Mod("Data/Data_Encodee_V2_2_2020.csv") 
        
    if "df_2020" not in st.session_state :
        st.session_state["df_2020"] = Data_Mod("Data/Data_Encodee_V2_2_2020.csv")
    
    
    
    # Préparation des données pour la modélisation.
# Suppression de certaines colonnes
    @st.cache_data 
    def donnes_modelisation() :

        """""
        Préparation des données pour les modeles de reg
        """""
        
        df_2020_mod = st.session_state["df_2020"].drop(columns = ["IncidentNumber","DateOfCall","TimeOfCall","PropertyType","AddressQualifier",
                            "Postcode_full","Postcode_district","IncidentStationGround","Easting_m","Northing_m","Easting_rounded","Northing_rounded","Latitude",
                            "Longitude","Latitude_Station","Longitude_Station","NumStationsWithPumpsAttending","NumPumpsAttending","PumpCount",
                            "PumpMinutesRounded","Notional Cost (£)","NumCalls","FirstPumpArriving_TravelTimeSec",
                            "FirstPump_DelayCode_Description","FirstPump_Division_staion","tempsAPI"], axis = 1)

        df_2020_mod = df_2020_mod.dropna(axis=0)

        # Features 
        X = df_2020_mod.drop(columns=["Weekday","Month","HourOfCall","Week_Weekend","London_Zone","CalYear","Same_Incident_Station",
                                "FirstPumpArriving_AttendanceTime","AttendanceTime_Min","Periode","Periode_Rush","FirstPump_Delayed","FirstPumpArriving_TurnoutTimeSec",
                                "Station_DelayFreq","IncGeo_WardNameNew","Ward_DelayFreq","Bo_DelayFreq" ,'Incident_Fire', 'Incident_Special Service',
            'StopCode_Primary Fire', 'StopCode_Secondary Fire',"IncidentGroup",
            'StopCode_Special Service', '_Non Residential',"_Other Vehicle", '_Other Residential',
            '_Outdoor', '_Outdoor Structure', '_Road Vehicle'])

        # y pour la regression
        y_reg = df_2020_mod.FirstPumpArriving_AttendanceTime

        # y pour la classification
        y_class = df_2020_mod.AttendanceTime_Min

        # encodage de la target
        y_class = y_class.replace({'0-3min' : 0,
                '3-6min' : 1,
                '6-9min' : 2,
                "9-12min" : 3,
                '+12min' : 4
                })
        
        return X, y_reg, y_class

    X_L , y_reg, y_class =  donnes_modelisation()

    if "X_L" not in st.session_state :
        st.session_state["X_L"] = donnes_modelisation()[0]

    if "y_reg" not in st.session_state :
        st.session_state["y_reg"] = donnes_modelisation()[1]

    if "y_class" not in st.session_state :
        st.session_state["y_class"] = donnes_modelisation()[2]
        st.title("Modélisation")
        
    tab1, tab2 = st.tabs(["Régression", "Classification 1"])
    
    with tab1 :
        # charge des moedeles de regression et scalers
        @st.cache_resource
        def load_model_reg():
            Elastic = pickle.load(open("ModelesLineaire/Elastic", 'rb'))

            return Elastic
        
        Elastic = load_model_reg()
        
        st.header("Prédiction des temps de réponse avec des modèles de régression")
        st.markdown("""
        Une première approche a été d'utilser des modèles de régression dans le but de prédire le temps de réponse avec précision.
        """)
        st.subheader("1. Performance du modèle")
        # split pour la regression
        X_train_reg,X_test_reg,y_train_reg, y_test_reg = train_test_split(X_L,y_reg, test_size = 0.25, random_state=42)

        # encoadge
        mean_enc_reg = MeanEncoder(smoothing='auto')
        X_train_reg = mean_enc_reg.fit_transform(X_train_reg,y_train_reg)
        X_test_reg = mean_enc_reg.transform(X_test_reg)

        scaler_reg = RobustScaler()
        X_train_reg_sc = scaler_reg.fit_transform(X_train_reg)
        X_test_reg_sc = scaler_reg.transform(X_test_reg)

        
        modeles_reg = ["ElasticNet"]

        dico_model_reg = { "ElasticNet" : Elastic }
        
        # ne pas cacher
        def calcul_metrics_reg(model,X_train,X_test):
            """""
            Calcul des metrics des modeles de régression
            """""
            y_pred_train = model.predict(X_train)
            R2_train = round(r2_score(y_train_reg, y_pred_train),3)
            MAE_train = round(mean_absolute_error(y_train_reg,y_pred_train),3)
            MSE_train = round(mean_squared_error(y_train_reg,y_pred_train),3)

            # valeur pour le set de test
            y_pred_test = model.predict(X_test)
            R2_test = round(r2_score(y_test_reg, y_pred_test),3)
            MAE_test = round(mean_absolute_error(y_test_reg,y_pred_test),3)
            MSE_test = round(mean_squared_error(y_test_reg,y_pred_test),3)
            
            return R2_train, MAE_train, MSE_train, R2_test, MAE_test, MSE_test

        
       
       # conserver en mémoire les valeurs pour chaque modeles

        # crée une valeur de dico model choisie contenant dico avec nom de modeles et leurs métriques
        if "model_reg_choisi" not in st.session_state :
            st.session_state["model_reg_choisi"] = {}
        
        
        # if "Reg_model_1" not in st.session_state : 
        #     st.session_state.Reg_model_1 = None

        # col1,col2 = st.columns(2)
        # with col1 :
        reg_model1 = st.selectbox("Sélectionnez un modèle", modeles_reg, key = "reg_mod1")
        md_reg1 = dico_model_reg[reg_model1]

        if reg_model1 in st.session_state["model_reg_choisi"] :
            R2_train, MAE_train, MSE_train, R2_test, MAE_test, MSE_test, y_pred = st.session_state["model_reg_choisi"][reg_model1].values()
        
        else  :
            R2_train, MAE_train, MSE_train, R2_test, MAE_test, MSE_test = calcul_metrics_reg(md_reg1,X_train_reg_sc,X_test_reg_sc)
            
            y_pred = md_reg1.predict(X_test_reg_sc[0:250,:])
        
        st.write("R2 train :", R2_train, " R2 test :", R2_test )
        st.write("MAE train :" , MAE_train," MAE test :",MAE_test)
        st.write("MSE train :", MSE_train," MSE test :" ,MSE_test)
        st.write("Différence MSE train - MSE train : " , round((MSE_train - MSE_test),3))
        
        if reg_model1 not in st.session_state["model_reg_choisi"] :
            st.session_state["model_reg_choisi"][reg_model1] = {"R2_train" :R2_train, "MAE_train" : MAE_train, "MSE_train" :MSE_train,
                                                            "R2_test" :R2_test, "MAE_test" : MAE_test, "MSE_test" :MSE_test, "y_pred" : y_pred }

        
        
        st.text(" \n")
         
        y_pred1 = md_reg1.predict(X_test_reg_sc[0:250])
        fig = plt.figure(figsize = (15,8))
        plt.plot(y_pred1)
        plt.plot(y_test_reg[0:250].values)
        plt.legend(["predit","reel"])
        plt.title("Comparaison des valeurs prédites et réelles")
        plt.xlabel("Index")
        plt.ylabel("Temps de réponse (sec)")
        st.pyplot(fig)
        
        
        fig = plt.figure(figsize = (15,8))
        # plt.plot(y_pred2)
        ax = sns.histplot(y_pred1, color="red",kde=True)
        sns.histplot(y_test_reg[0:250].values, ax=ax, color="blue",kde=True)
        # plt.plot(y_test_reg[0:250].values)
        plt.legend(["predit","reel"])
        plt.xlabel("Temps de réponse (en sec)")
        plt.title("Comparaison des valeurs prédites et réelles")
        st.pyplot(fig)
        
        st.text(" \n")
        st.markdown("""
        Les modèles sont assez précis pour prédir les valeurs intermédaires (MAE de 60 secondes). Mais lorsque les valeurs réeles se rapprochent des extrêmes,
                     l'écart entre les prédictions et les valeurs réeles augmente fortement.
        """)


        st.text(" \n")
        st.subheader("2. Interprétation et Prédictions")
        st.text(" \n")
        import shap.explainers
        # pour visualisation des plots
        shap.plots.initjs()

        explainer_lg = shap.explainers.LinearExplainer(Elastic, X_test_reg_sc,feature_names=X_L.columns)
        shap_values_lg = explainer_lg(X_test_reg_sc)
        
    
        # # plot des features importances
        @st.cache_data
        def sum_plot_elastic() :
            fig = plt.figure()
            shap.summary_plot(shap_values_lg, X_test_reg_sc, plot_type="bar",plot_size=[11,6])
            return fig
        
        st.pyplot(sum_plot_elastic())
        st.text(" \n")
        st.text(" \n")
        st.text(" \n")
        """
        ###### La distance est la variable qui impact le plus les prédictions du modèle.  
        ###### On décide de conserver les autres variables car elles peuvent avoir un impact sur la précision de la prédiction
        """
        st.text(" \n")
        st.text(" \n")
    
        st.markdown("Le plot ci dessous permet de voir pour un incident n, comment les valeurs des variables de ont impacté la prédiction moyenne du modèle")
        # # redonne aux variables leur valeur d'origine
        X_test_reg = mean_enc_reg.inverse_transform(X_test_reg)
        shap_values_lg.data = X_test_reg.values
        
        if "select_pred_reg" not in st.session_state :
            st.session_state["select_pred_reg"] = 0

        # permet de générer une valeur aléatoire et de faire le plot sans rerun le script à chaque
        @st.experimental_fragment
        def generate_val():
            generate_value_reg = st.button("Prédiction pour un incident")
            if generate_value_reg :
                st.session_state["select_pred_reg"] = np.random.randint(0,len(X_test_reg))
                st.write("valeur" , st.session_state["select_pred_reg"])
            fig = plt.figure()
            shap.plots.waterfall(shap_values_lg[st.session_state["select_pred_reg"]])
            st.pyplot(fig)

        generate_val()
        

        ### remplir des champs pour prédictions
        st.text(" \n")
        
        st.markdown("  **Génération d'une prédiction de temps de réponse**  \n")

        @st.experimental_fragment
        def generate_prediction():
                
                pred_model_reg = st.selectbox("Sélectionnez un modèle", modeles_reg, key = "reg_model_pred")
                md_pred_reg = dico_model_reg[pred_model_reg]
                col1 , col2 = st.columns(2)

                with col1 : 
                    mois = st.number_input("Choisir un mois", min_value=0, max_value=12, step=1 , key = "month")
                    jour = st.number_input("Choisir un jour", min_value=0, max_value=7, step=1, key = "day")
                    quartier = st.selectbox("Sélectionnez un quartier", X_L["IncGeo_BoroughName"].unique(), key = "borought")
                    StopCode = st.selectbox("Sélectionnez l'urgence", X_L["StopCodeDescription"].unique(), key = "urgence")

                with col2 :        
                    heure = st.number_input("Choisir une heure", min_value=0, max_value=24, step=1, key ="hour")
                    distance = st.number_input("Choisir une distance (en km)", min_value=0.5,step=0.5 , key = "dst", format="%.2f")
                    Station_deployee = st.selectbox("Sélectionnez la station déployée", X_L["FirstPumpArriving_DeployedFromStation"].unique(), key = "station")
                    Propriete = st.selectbox("Sélectionnez la propriété touchée", X_L["PropertyCategory"].unique(), key = "prop")

                SST = st.selectbox("Sélectionnez le service special impliqué", X_L["SpecialServiceType"].unique(), key = "sp_serv")

                # ajouste les données temporelles aux bonnes valeurs
                heure_cos = np.cos(2*np.pi * heure/24)
                heure_sin = np.sin(2*np.pi * heure/24)
                mois_cos = np.cos(2*np.pi * mois/12)
                mois_sin = np.sin(2*np.pi * mois/12)
                jour_cos = np.cos(2*np.pi * jour/7)
                jours_sin = np.sin(2*np.pi * jour/7)


                var_prediction = pd.DataFrame(np.array([[mois_cos, mois_sin, jour_cos,jours_sin, heure_cos,heure_sin, SST,quartier,Station_deployee,distance,StopCode,Propriete ]]),
                                            columns = X_test_reg.columns)
                
            
                var_prediction = mean_enc_reg.transform(var_prediction)
                var_prediction_sc = scaler_reg.transform(var_prediction)
                

                temps = md_pred_reg.predict(var_prediction_sc)[0]
                
                st.write("Temsps de réponse prédit : ",temps//60,"min", round(temps%60,0),"sec")

        generate_prediction()
        st.text(" \n")
        st.markdown("**Dans les prédictions faites par le modèle, la distance va définir le temps de réponse globale, les autres variables vont permettre d'affiner la prédiction au niveau des secondes**")



    # classification bins de 3 mins
    with tab2 : 
        
        
         # chargement des modèles entrainné et scaler
        @st.cache_resource
        def load_model_class():
            RandomForest_class = pickle.load(open('ModeleClass2/RandomForest', 'rb'))
            # mean_enc_class = pickle.load(open("Scalers/MeanEncoder_class", 'rb'))
            # scaler_class = pickle.load(open("Scalers/RbScaler_class", 'rb'))
            
            return RandomForest_class
        
        RandomForest_class = load_model_class()
        
        st.header("Prédiction des temps de réponse avec des modèles de classification")
        st.markdown("""
        Pour la classification, les temps de réponse ont été découpé en classes de 3 minutes plus une classe pour les temps dépassant 12 minutes     
        """)
    
        st.subheader("1. Performances de modèle")
        st.text(" \n")
        

        # Split pour la classification
        X_train_class,X_test_class,y_train_class,y_test_class = train_test_split(X_L,y_class, test_size = 0.25,random_state=42)

        mean_enc_class = MeanEncoder(smoothing='auto')
        X_train_class = mean_enc_class.fit_transform(X_train_class,y_train_class)
        X_test_class = mean_enc_class.transform(X_test_class)

        scaler_class = RobustScaler()
        X_train_class_sc = scaler_class.fit_transform(X_train_class)
        X_test_class_sc = scaler_class.transform(X_test_class)


    # # pour arranger l'ordre des colonnes et lignes dans les rapports de class et mat de conf
        dico = {0 : "0-3min"
            ,1 : "3-6min",
            2 : "6-9min",
            3 : "9-12min",
            4 : "+12min"}
        order = ["0-3min", "3-6min", "6-9min","9-12min","+12min"]
        
        modeles = ["RandomForest"]

        # pour récupérer le model à partir de leur nom
        dico_model_class = { "RandomForest": RandomForest_class,
                   
                   }
        
        #fonction qui affiche le rapport de classif et la mat de conf
        def rapport_mat(model) : 
            clf = dico_model_class[model]
            y_pred_test = clf.predict(X_test_class_sc)
            y_pred_adj = [dico[i] for i in y_pred_test]
            mat = pd.crosstab(y_test_class.replace(dico),y_pred_adj,rownames=["reels"],colnames=["predits"]).reindex(index=order ,columns=order)
            rapport = pd.DataFrame(classification_report_imbalanced(y_test_class,y_pred_test, target_names = order, output_dict = True))
            
            return (mat ,rapport)

        def calcul_metrics_class(model) :
            y_pred_train = model.predict(X_train_class_sc)
            blc_acc_train = round(balanced_accuracy_score(y_train_class,y_pred_train),3)
            f1_train = round(f1_score(y_train_class,y_pred_train, average="weighted"),3)
            geo_mean_score_train = round(geometric_mean_score(y_train_class,y_pred_train,average="weighted"),3)

            y_pred_test = model.predict(X_test_class_sc)
            blc_acc_test = round(balanced_accuracy_score(y_test_class,y_pred_test),3)
            f1_test = round(f1_score(y_test_class,y_pred_test, average="weighted"),3)
            geo_mean_score_test = round(geometric_mean_score(y_test_class,y_pred_test,average="weighted"),3)
            
            return blc_acc_train,f1_train,geo_mean_score_train,blc_acc_test,f1_test,geo_mean_score_test


        # crée une valeur de dico model choisie contenant dico avec nom de modeles et leurs métriques
        if "model_class_choisi" not in st.session_state :
            st.session_state["model_class_choisi"] = {}
        
            
    
        class_option1 = st.selectbox("Sélectionnez un modèle", modeles, key = "class_mod1")
        mat, rapport = rapport_mat(class_option1)
        st.markdown("**Matrice de confusion (réels en ligne, prédits en colonne)**")
        st.table(mat)
        st.markdown("**Rapport de classification**")
        st.dataframe(rapport.transpose()[0:5],width=450)
        st.markdown("**moyennes des métriques**")
        st.dataframe(rapport.iloc[0:1,5:])

        md_class_1 = dico_model_class[class_option1]
        blc_acc_train,f1_train,geo_mean_score_train,blc_acc_test,f1_test,geo_mean_score_test = calcul_metrics_class(md_class_1)
        st.write("balanced accuracy train :" , blc_acc_train , "balanced accuracy test :" , blc_acc_test)
        st.write("F1 train :" , f1_train, "\n", "F1 test :" , f1_test)
        st.write("geometric mean train :" , geo_mean_score_train,"geometic mean test :" , geo_mean_score_test)

        if class_option1 not in st.session_state["model_class_choisi"] :
            st.session_state["model_class_choisi"][class_option1] = {"balanced_accuracy_train" :blc_acc_train, "F1_train" : f1_train, "geometric_mean_train" :geo_mean_score_train,
                                                            "balanced_accuracy_test" :blc_acc_test, "F1_test" : f1_test, "geometric_mean_test" : geo_mean_score_test }


        st.text(" \n")
        st.text(" \n") 
        st.subheader("2.Prédictions")
        st.text(" \n")
        st.text(" \n")
        
        # interprétation shap
        # explainer_class = shap.Explainer(RandomForest_class,feature_names=X_train_class.columns)
        # shap_values_cat = explainer_class(X_test_class_sc)
        
        # X_test_scaled = pd.DataFrame(X_test_class_sc,columns=X_train_class.columns)

        # shap_values_class = explainer_class(X_test_scaled)  
        
        # remet les valeurs
        # X_test_origine = mean_enc_class.inverse_transform(X_test_class)
        # shap_values_cat.data = X_test_origine.values
        
        # # plot des features importances
        # @st.cache_data
        # def sum_plot_RF() :
        #     fig = plt.figure()
        #     shap.summary_plot(shap_values_cat, X_test_class_sc, plot_type="bar",plot_size=[11,6])
        #     return fig
        
        # def bar_plot_RF() : 
        #     fig = plt.figure()
        #     shap.plots.bar(shap_values_cat)
        #     return fig

        
        # st.pyplot(sum_plot_RF())
        st.text(" \n")
        st.text(" \n")
        st.text(" \n")
        # st.pyplot(bar_plot_RF())
        
        # st.write(shap_values_cat.shape)
        # classes = ["0-3min","3-6min","6-9min","9-12min","+12min"]

        # for i,classe in zip(range(0,5), classes) :
        #     print("probilité d'arriver en",classe,"pour un individu n")
            
        #     shap.plots.waterfall(shap_values_class[3,:,i],show = False)
        #     plt.title(f"probilité d'arriver en {classe} pour un incident donné")
        #     plt.show()
        
        
        

        st.markdown("**Génération d'une prédiction de temps de réponse**")

        @st.experimental_fragment
        def generate_prediction():
            model_class_pred = st.selectbox("Sélectionnez un modèle", modeles,key = "mod_class_pred")
            prediction_md = dico_model_class[model_class_pred]
            
            col1 , col2 = st.columns(2)

            with col1 : 
                mois = st.number_input("Choisir un mois", min_value=0, max_value=12, step=1 , key = "class_month")
                jour = st.number_input("Choisir un jour", min_value=0, max_value=7, step=1, key = "class_day")
                quartier = st.selectbox("Sélectionnez un quartier", X_L["IncGeo_BoroughName"].unique(), key = "class_borought")
                StopCode = st.selectbox("Sélectionnez l'urgence", X_L["StopCodeDescription"].unique(), key = "class_urgence")

            with col2 :        
                heure = st.number_input("Choisir une heure", min_value=0, max_value=23, step=1, key ="class_hour")
                distance = st.number_input("Choisir une distance (en km)", min_value=0.5,step=0.5 , key = "class_dst",format = "%.2f")
                Propriete = st.selectbox("Sélectionnez la propriété touchée", X_L["PropertyCategory"].unique(), key = "class_prop")
                Station_deployee = st.selectbox("Sélectionnez la station déployée", X_L["FirstPumpArriving_DeployedFromStation"].unique(), key = "class_station")

            SST = st.selectbox("Sélectionnez le service special", X_L["SpecialServiceType"].unique(), key = "class_sp_serv")

            # ajouste les données temporelles aux bonnes valeurs
            heure_cos = np.cos(2*np.pi * heure/24)
            heure_sin = np.sin(2*np.pi * heure/24)
            mois_cos = np.cos(2*np.pi * mois/12)
            mois_sin = np.sin(2*np.pi * mois/12)
            jour_cos = np.cos(2*np.pi * jour/7)
            jours_sin = np.sin(2*np.pi * jour/7)


            var_prediction = pd.DataFrame(np.array([[mois_cos, mois_sin, jour_cos,jours_sin, heure_cos,heure_sin, SST,quartier,Station_deployee,distance,StopCode,Propriete ]]),
                                        columns = X_test_reg.columns)
            
            var_prediction = mean_enc_reg.transform(var_prediction)
            var_prediction_sc = scaler_reg.transform(var_prediction)

            pred = prediction_md.predict(var_prediction_sc)[0]
            temps =dico[pred]
            # prends l'argument du dico pour renvoiyer la classe
            # dico = {0 : "0-3min"
            # ,1 : "3-6min",
            # 2 : "6-9min",
            # 3 : "9-12min",
            # 4 : "+12min"}

            st.write("Temsps de réponse prédit :" , temps)

        generate_prediction()

       

         

 

