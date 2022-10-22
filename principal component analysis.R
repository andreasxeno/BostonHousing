# Preliminaries ================
setwd("~/OneDrive - University of New Haven/Spring 2022/BANL 6420-Unsupervised Machine Learning/Week 4 2.14.2022")
options(digits = 3, scipen = 9999)
remove(list = ls())

#' Load the packages
      suppressPackageStartupMessages(library(factoextra))
      suppressPackageStartupMessages(library(FactoMineR))
      suppressPackageStartupMessages(library(dplyr))
      suppressPackageStartupMessages(library(psych))
      suppressPackageStartupMessages(library(naivebayes))
      suppressPackageStartupMessages(library(MASS))
      suppressPackageStartupMessages(library(DescTools))
      suppressPackageStartupMessages(library(GGally))
      suppressPackageStartupMessages(library(ggthemes))
      suppressPackageStartupMessages(library(forecast))
      suppressPackageStartupMessages(library(caret))
      suppressPackageStartupMessages(library(ggbiplot))
      suppressPackageStartupMessages(library(ggrepel))
      suppressPackageStartupMessages(library(randomForest))
      
# Introduction ===============================
      cnbc = read.csv("cnbc_data_2021-1.csv", header = TRUE)
      cnbc_df = read.csv("cnbc_data_2021-1.csv", header = TRUE)
      cnbc = cnbc %>% dplyr::select(-OVERALL) %>% tibble::column_to_rownames("State")
      
      fviz_cluster(kmeans(cnbc, 3), data = cnbc)
      
      
      fviz_screeplot(cnbc.pca, addlabels = T)
      cnbc.pca = PCA(cnbc, ncp = 3, graph = TRUE)
      names(cnbc.pca) #always use this command to understand these variables 
      cnbc.pca$eig
      cnbc.pca$var
      fviz_screeplot(cnbc.pca, addlabels = T)
      fviz_pca_var(cnbc.pca)
      
      
      fviz_pca_biplot(cnbc.pca, 
                      repel = TRUE,
                      col.var = "black",
                      col.ind = cnbc_df$State, 
                      legend.title = "Best Place for Business", 
                      addEllipses = TRUE,
                      max.overlaps = 25) + 
        theme(legend.position = "none")
    #x-axis is the first pca component
      #y-axis is the second pca component
      #to better understand the graph: massachussets are mostly associated with high cost of living and high cost 
      #doing business. 
#' 
## ------------------------------------------------------------------------ #
# Load the data ================================== 
      data(Boston)
      Bos = Boston
        names(Bos)
          str(Bos)
            head(Bos)
#measuring the accuracy statistic. the data reduction performs less efficiently than the actual data.      
#' 
#' The left hand side variable is medv (median value) - the value of the property.
#' But for a factor variable identifying proximity to the Charles River (chas), 
#' all explanatory variables are continuous.
#' 
#' Check to see if there are any na's

      colSums(is.na(Bos))
#' 
#' Examine their correlation
#' 
              Bos %>% dplyr::select(-medv) %>% ggcorr(label = TRUE)

              #high correlation, there is redundancy in the data, see color red.
              
#' Take a look at medv. We will predict housing prices (medv) 
#' after we build our unsupervised model.
#' 

#'   PCA Modeling
#'   1. Extract the components and determine the number to retain
#'   4. Create scores from the components
#'   5. Use the scores as input variables for regression analysis and
#'       evaluate the performance. 
#' 
#' There are several packages that contain principal components. 
#' We use here
#' the package psych.
#' 
#' 
# Extract the Principal Components ==================
              Bos_df = Bos %>% dplyr::select(-medv)
              
              Bos_pca = principal(Bos_df, 
                                  normalize = TRUE, 
                                        scores= TRUE, 
                                          rotate = "none")
#' 
#' A scree plot displays the eigenvalues associated with each of the PCs.
#' 
#' Look for the "bend" or "knee."  In addition, see where the eigenvalues become less than one.
#' 
#' Any eigenvalue greater than 1 suggests that the associated PC explains more
#' variation.
#' 
              names(Bos_pca)

              plot(Bos_pca$values, type = "b", ylab = "Eigenvalues", xlab= "Component")
            
# its a four by looking at the scree plot' 
#' Let's go with 3 or 4; another rule of thumb, chose the number of eigenvalues that explain
#' 70-80 percent of the variation.

              sum(Bos_pca$values[1:3])/sum(Bos_pca$values)
              sum(Bos_pca$values[1:4])/sum(Bos_pca$values)
#' 
#' Rerun the command with 4 nfactors to extract the components

        pca_4 = principal(Bos_df, 
                          nfactors = 4, 
                              rotate = "none",
                                   normalize = TRUE)
                
                pca_4
#' use the four pca_4 to estimate the pricing model, see regression below
#' 
#' Now run a linear regression with LM Prices explained by the selected three PC's.

        lm_fit <-  lm(Bos$medv ~., data = as.data.frame(pca_4$scores))
                  
                summary(lm_fit) 
               
#' Predict prices (medv) and display actual (medv) vs. predicted median prices (lmpred)
#'                 

        Bos$lmpred <- predict(lm_fit)
                
                ggplot(Bos, aes(x = medv, y = lmpred)) +
                  geom_point() +
                  geom_smooth(method = "lm", se = FALSE)  +
                  theme_classic() +
                  labs(title =" Regression with 4 Principal Components", 
                       x = "Actual Median Prices",
                       y = "Predicted Median Prices")
                               
#there is errors,so we will use the measure of accuracy' 
#' Examine performance of model using the function accuracy from the package Forecast.
#' Make a note of the RMSE.

        accuracy(Bos$lmpred,Boston$medv)
#' 
#' FYI: there is no need to "predict" after LM; one of the objects it creates are the
#' predicted values in the object: "fitted.values"

        accuracy(lm_fit$fitted.values, Boston$medv )

#' 
#' 
#' Now we repeat the exercise but using all the explanatory variables 
#' in the model instead of the selected principal components.
      Bos = Boston

      lm_2 = lm(Bos$medv~., data = Bos)

#' 
#' And in turn, examine the performance with the function accuracy();
#' we look for RMSE.
#' 
#' Notice the lower RMSE which suggests a better fit than the PC model.
#' 
#' 
      lm2_pred = predict(lm_2, Bos)
      
      accuracy(lm2_pred, Bos$medv)
#'       
#' Lets look at it visually. 
#' But first i have to attach the predicted prices to the data set.
#' 

      Bos$pred2 = predict(lm_2, data = Bos)

      ggplot(Bos, aes(x = medv, y = pred2)) +
                  geom_point() +
                  geom_smooth(method = "lm", se = FALSE)   +
        theme_classic() +
        labs(title =" Regression with Full Data Set", 
             x = "Actual Median Prices",
             y = "Predicted Median Prices")
      

#new dataset below'                 
# Principal Components and Classification Models =====================
#' 
#' Here we use financial variable to predict failure in the tradition of
#' Altman and Ohlsen.
#' 

      fail = read.csv("failed_c_V2.csv", header = T, check.names = T)
      head(fail)

 
# Variable 	  Description
# CF_TD	      X1=(Cash Flow)/(Total Debt)
# NI_TA       X2=(Net Income)/(Total Assets)
# CA_CL       X3=(Current Assets)/(Current Liabilities)
# CA_NS       X4=(Current Assets)/(Net Sales)
# after_2_yrs	Status after 2 years
# Failed	    Failed within 2 Years
# 
# Observation:  Failed and status after 2 years appear to be identical
# 

      table(fail$after_2_yrs, fail$Failed)
      Y_Failed = fail$Failed
      fail_df = fail %>% dplyr::select(-Failed, -after_2_yrs)

#' 
#' Examine extent of correlation                

            ggcorr(fail_df, label = TRUE)

#' 
#' Extract PC and examine scree plot                

      fail_pc = principal(fail_df, normalize = TRUE, rotate = "none")
      
      plot(fail_pc$values, type = "b", ylab = "Eigenvalues", xlab = "Component")

#2 form the scree plot'                 
#' Go with two PC and rerun model                

      fail_pc2 = principal(fail_df, 
                           normalize = TRUE, 
                            rotate = "none", 
                              nfactors = 2)

#'                 
#' Now use the two PC to predict Failure              
#' First use predict() and then convert to a dataframe
#' The use Naive Bayes to obtain classification algorithm (it helps us classify the algorithm)
#' 
#' Use fitted model to predict classification
#' 

      pred.Failed = predict(fail_pc2, data = fail_df)
      pred.Failed = as.data.frame(pred.Failed)
      mod_pc = naive_bayes(as.factor(Y_Failed)~., data = pred.Failed)
      mod_pc_pred = predict(mod_pc, data = pred.Failed)

#'               
#' Then use the function confusionMatrix from the package caret 
#' to appraise the quality of the classification
#' 
#' As you can see - the result when using PCs as explanatory variable shows an 87% classifaction accuracy              
#' 

      caret::confusionMatrix(as.factor(mod_pc_pred), as.factor(Y_Failed))

#' see matrix, 22+18/total number, thats how we find the 0.87 accuracy. we have a total of 6 mistakes (3+3)
#' Now we repeat the exercise using all the variables instead of the PC.
#' 

      mod_nb = naive_bayes(as.factor(Y_Failed)~., data = fail_df)
      
      mod_nb_pred = predict(mod_nb, data = fail_df)

#' 
#' and again, we examine the accuracy. 
#' 
#' It is a slightly better 91.3% . 4 mistakes (3+1)
#' 

      confusionMatrix(as.factor(mod_nb_pred), as.factor(Y_Failed))
#'                 
#'  moral of the story: if we use more data, pca its better. the point of the exercise is to show whats coming next
#'  
# FactoMineR and factoextra ========================
#' We will use the great visuals from the package FactoMineR.
#' First we have to run the package's principal components function (PCA).
#' 

      bos.pca = PCA(Bos_df)
      fviz_pca_var(bos.pca)

#explanatory variables in the pca component' 
#' and for the fail dataset

      
      fail.pca = PCA(fail_df, graph = TRUE)
      fviz_screeplot(fail.pca, addlabels = T)
      fviz_pca_var(fail.pca)

#it handles both.' 
#' 
#' 

      fviz_pca_var(bos.pca, col.var = "cos2", 
                   gradient.cols = c("red", "blue","black"),
                   repel = TRUE)

#' 

      fviz_pca_var(fail.pca, col.var = "cos2", 
             gradient.cols = c("red", "blue","black", "orange"),
             repel = TRUE)

#' 
#' Graphs of Contributions
#' 
#' 
      fviz_contrib(bos.pca, choice = "var",
                      axes = 1:2)

#' 

      fviz_contrib(fail.pca, choice = "var",
                      axes = 1:2)

#' 
#' 

      fviz_pca_biplot(fail.pca, repel = TRUE,
                col.var = "blue",
                col.ind = "red")

#not that insightful. whats the second dimension. ' 

      fviz_pca_biplot(bos.pca, repel = TRUE,
                col.var = "blue",
                col.ind = "red")
      #you have the perceptual map. it explains the most components to interpret.
      #you need to understand whats going on. 
      #' 
      ## ---------------------------------------------------------------------------------------------------------------------------------------------#' 

      #' Using Principal Components to inform Brand Positioning Strategy
      #' Using brand_data.csv conduct an analysis of brand attributes to draw a
      #' perceptual map: aka principal components biplot.
      brand = read.csv("brand_data.csv", header = TRUE)
      
      head(brand)
      
      brand_df  = brand %>% dplyr::select(-brand)
      
      
      brand.pca = PCA(brand_df)
      fviz_pca_var(brand.pca, col.var = "darkred")
      fviz_pca_biplot(brand.pca, 
                      repel = TRUE, 
                      col.var = "darkred",
                      label = "var")
      
      #' Use this map to infer brand position understanding and strategy.
      #' re-run it with less brands, maybe clean it up. remember to clean it up 
      #' and run it 
    
      
      fviz_pca_biplot(brand.pca, 
                      repel = TRUE,
                      col.var = "black",
                      col.ind = brand$brand, 
                      legend.title = "Brand Map", 
                      addEllipses = TRUE,
                      max.overlaps = 40,
                      caption = "Andreas X")
    
    