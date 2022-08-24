import numpy as np

def generar_datos(nClusters, nDim, nDatos=50, molesto=False, sigma_max=5):
    """
    Función para generar aleatorios. Parámetros:
    nClusters: num clusters
    nDim: dimensión del espacio
    nDatos: cantidad de puntos por cluster
    molesto: default=False. Superpone el primer y último cluster
    sigma max
    """
    assert nClusters > 1, "Pasame más de un cluster"
    r = 30          #Tamaño del sistema
    scale = 2       #Distancia del cluster molesto al primer cluster

    data = np.zeros((nClusters*nDatos, nDim+1), dtype=float)    #Agrego el label del cluster al final

    mean = np.random.uniform(-r + 5, r - 5, size=(nClusters, nDim))

    if(molesto):
        cluster_molesto = mean[0,:] + scale*np.ones_like(mean[0,:])
        mean[-1,:] = cluster_molesto  #Reemplazo último cluster por cluster molesto
    sigma = np.random.uniform(.5, sigma_max, size=(nClusters))

    for p in range(nClusters):
        data[p*nDatos:(p+1)*nDatos,:-1] = np.random.normal(mean[p,:], sigma[p], size=(nDatos, nDim))
        data[p*nDatos:(p+1)*nDatos,-1] = p

    return data[:,:-1], data[:,-1]

