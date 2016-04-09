## Recommender System (Item Based - KNN)

### List of the files

**LaunchEC2_Romain.ipynb** : Ipython script leading the creation of the EC2 and the upload of the various files

**Romain_KNN_CF_Classical.ipynb** : Ipython script that implements the classical version of the algorithm which consists in computing the whole support, and matrice of similarity before actually make the prediction.

**Romain_KNN_CF_Optimized.ipynb** : Ipython script that implements a more *lazy* algorithm consisting in only computing the support and the similarity matrix based on the (user, artist) duo. 

### Algorithm

The idea consists in making a prediction by adjusting the classical baseline model with a collaborative filtering approach with consists in using the K closest neighbors (KNN) to the considered item (m) and weight the bias by the similarity.
$$
\hat{Y_{um}} = \hat{Y_{um}^{Baseline}} + \frac{\sum_{j \in S^k(m)} s_{mj} (Y_{uj} - \hat{Y_{um}^{Baseline}})}{\sum_{j \in S^k(m)} s_{mj}}
$$

The similarity function can be computed as :
$$
s_{mj} = \frac{N_{Common} \rho_{mj}}{N_{Common} + reg}
$$
And we here remind the clasical baseline model : 
$$
\hat{Y_{um}^{Baseline}} = \overline{Y} + (\overline{Y_u} - \overline{Y}) + (\overline{Y_m} - \overline{Y})
$$





