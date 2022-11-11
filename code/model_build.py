# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:24:59 2022

@author: Pierre
"""

import pandas as pd
import numpy as np
import re

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline 
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import GradientBoostingClassifier


class AgeCat(BaseEstimator, TransformerMixin):
    # BaseEstimator contient les méthodes get_params et set_params.
    # TransformerMixin contient la méthode fit_transform.
    
    def __init__(self):
        return None   
        
    def fit(self, X, y = None):
        self.maxAge = np.max(X.Age)
        return self
    
    def transform(self, X): # Création de la nouvelle colonne 

        return pd.DataFrame(pd.cut(X['Age'],
                                   bins = [0, 12, 18, 30, 50, 65, self.maxAge],
                                   labels=['Kid', 'Adolescent', 'Adult-', 'Adult', 'Adult+', 'Senior']))
    
class FamilySize(BaseEstimator, TransformerMixin):
    # BaseEstimator contient les méthodes get_params et set_params.
    # TransformerMixin contient la méthode fit_transform.
    
    def __init__(self):
        return None

        
    def fit(self, X, y = None):  # Ne fait rien. 
        return self
    
    def transform(self, X): # somme toutes les personnes de la famille + lui même 
        return pd.DataFrame(X[['SibSp', 'Parch']].sum(axis=1)+1)
    
    
def splitter(X):
    
    ''' Cette fonction auxiliaire utilise les expressions régulières afin de créer 
    une liste de 3 éléments pour chaque élément d'une colonne de DataFrame.  
    Construite pour la colonne 'Name', elle permettra de créer les listes qui auront 
    le nom, le titre et le(s) prénom(s)'''
        
    chaine=X.apply(lambda x: re.split("[.,\((.*?)\)]",x))

    for elt in chaine:
        if elt[-1]=='': # on enlève des chaines de caractères vides à la fin ''
            del(elt[-1])
    return(chaine)

class SplitName(BaseEstimator, TransformerMixin):
    
    def __init__(self, column_name):
        self.column_name = column_name   # nom de la colonne à segmenter
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        chaine=splitter(X[self.column_name]) # Retourne une liste de 3 éléments pour extraire le nom, le prénom et le titre 

        X['Surname'] = chaine.apply(lambda x : x[0] ) 
        X['Title'] = chaine.apply(lambda x : x[1][1:]) # Car il y a un espace
        X['Firstname(s)'] = chaine.apply(lambda x : x[2] )
     

        return X
    
def ReturnCabin(l):
    for elt in l:
        if type(elt) is str: 
            return elt 
    return np.nan


class AddCabins(BaseEstimator, TransformerMixin):

    def __init__(self, column_id, column_cabin, column_name):
        self.column_id = column_id 
        self.column_cabin = column_cabin #Nom de la colonne des Ids 
        self.column_name = column_name   # nom de la colonne à segmenter
        
        
    def fit(self, X, y=None):

        occurences = X[self.column_name].unique()
        self.Cabin_list = X[self.column_cabin].apply(lambda x: [])

        for family in occurences:
            liste = X[X[self.column_name] == family][self.column_cabin].tolist()
            for id, name  in zip(X[self.column_id], X[self.column_name]):
                if name == family:
                    self.Cabin_list.at[id-1] = liste
        return self
    
     
    def transform(self, X):
        X.loc[:,self.column_cabin] = self.Cabin_list.apply(ReturnCabin)
        
     
        return X
    
def babtri(x):
    if x%2==0.0:
        return('Babord')
    if x%2==1.0:
        return('Tribord')
    else:
        return np.nan


class SplitCabin(BaseEstimator, TransformerMixin):
    # BaseEstimator contient les méthodes get_params et set_params.
    # TransformerMixin contient la méthode fit_transform.
    
    def __init__(self, column_name):
        self.column_name = column_name   # nom de la colonne à segmenter
        
    def fit(self, X, y = None):  # Ne fait rien. 
        return self
    
    def transform(self, X): # Création de la nouvelle colonne 
        new_X = pd.DataFrame()

        new_X[self.column_name+'_letter'] = X[self.column_name].str.slice(0,1)
        var=X[self.column_name].str.slice(1,5).str.extract("([0-9]+)").astype("float") # variable qui permet d'avoir le numéro de la cabine 
        new_X[self.column_name+"_parite"] = (var.iloc[:,0].apply(babtri))

        return new_X
    
class CategorizeTitle(BaseEstimator, TransformerMixin):
    # BaseEstimator contient les méthodes get_params et set_params.
    # TransformerMixin contient la méthode fit_transform.
    
    def __init__(self, column_name):
        self.column_name = column_name   # nom de la colonne à segmenter
        
    def fit(self, X, y = None):  # Ne fait rien. 
        return self
    
    def transform(self, X): # Création de la nouvelle colonne 
        

        special = ["Dr", "Don","Mme","Major","Lady","Sir","Mlle","Col","Capt","the Countess","Jonkheer","Dona"]

        X[self.column_name] = X[self.column_name].replace(special, 'Special')
        X[self.column_name] = X[self.column_name].replace(['Rev'], 'Mr')
        X[self.column_name] = X[self.column_name].replace(['Ms'], 'Miss')
        
  

        return X[[self.column_name]]
    
    
def build_pipeline():
    complete_cabins = AddCabins('PassengerId','Cabin','Surname')
    cabin_split     = SplitCabin('Cabin')
    cabin_si        = SimpleImputer(strategy='constant', fill_value="missing")
    cabin_ohe       = OneHotEncoder()
    
    
    CabinsPipeline = Pipeline(steps=[('Complétion des cabines', complete_cabins),
                                     ('Séparation des cabines', cabin_split),
                                     ('Imputation cabines', cabin_si),
                                     ('Encodage cabines', cabin_ohe)])
    
    
    cat_title = CategorizeTitle('Title')
    title_si  = SimpleImputer(strategy='most_frequent')
    title_ohe = OneHotEncoder()
    
    
    TitlePipeline = Pipeline(steps=[('Catégorisation des titres', cat_title),
                                    ('Imputation titres', title_si),
                                    ('Encodage titres', title_ohe)])
    
    
    FeatureUnionPipeline = FeatureUnion([("Cabin", CabinsPipeline),
                                     ("Title", TitlePipeline)])
    
    name_split = SplitName('Name')

    NamePipeline = Pipeline([('Séparation du nom', name_split),
                             ('Feature Union', FeatureUnionPipeline)])
    
    size_family = FamilySize()
    size_si     = SimpleImputer(strategy='mean')
    size_st     = StandardScaler()
    
    SizeFamilyPipeline = Pipeline([('Taille famille', size_family),
                                   ('Imputation taille famille', size_si),
                                   ('Standardisation taille famille', size_st)])
    
    age_si          = SimpleImputer(strategy='most_frequent')
    age_ohe         = OneHotEncoder()
    age_categorized = AgeCat()
    
    AgePipeline = Pipeline([('Catégorisation des ages', age_categorized),
                            ('Imputation ages', age_si),
                            ('Encodage ages', age_ohe)])
    
    num = ['Pclass','Fare']
    cat = ['Sex', 'Embarked']
    
    num_si = SimpleImputer()
    num_st = StandardScaler()
    
    cat_si = SimpleImputer(strategy = 'most_frequent')
    cat_ohe = OneHotEncoder()
    
    
    NumericalPipeline = Pipeline(steps = [('valeurs_manquantes_num', num_si),
                                          ('standardisation', num_st)])
    
    CategorialPipeline = Pipeline(steps = [('valeurs_manquantes_cat',cat_si),
                                           ('encoder', cat_ohe)])
    
    preprocessor = make_column_transformer((NamePipeline, ['PassengerId','Name','Cabin']),
                                           (SizeFamilyPipeline,['SibSp','Parch']),
                                           (AgePipeline, ['Age']),
                                           (NumericalPipeline, num),
                                           (CategorialPipeline, cat))
    
    final_pipeline = Pipeline(steps= [('Feature Engineering', preprocessor), 
                                      ('Prediction', GradientBoostingClassifier())])
    
    return final_pipeline


    from model_train import model_train


    print(model_train())