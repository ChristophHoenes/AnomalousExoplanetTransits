import numpy as np
from sklearn.covariance import MinCovDet
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest


def mahalanobis_distance(codes, threshold=None):
    covariance_estimate = MinCovDet().fit(codes)
    mahal_dist = covariance_estimate.mahalanobis(codes - covariance_estimate.location_) ** (
            1 / (codes.shape[-1] + 1))
    if threshold is None:
        return mahal_dist, None, covariance_estimate
    else:
        outliers = mahal_dist > threshold
        return codes[outliers], np.nonzero(outliers)[0], covariance_estimate


def local_outlier_factor(codes, **lof_params):
    lof = LocalOutlierFactor(**lof_params)
    outliers = lof.fit_predict(codes) == -1
    return codes[outliers], np.nonzero(outliers)[0], lof


def one_class_svm(codes, **ocsvm_params):
    ocsvm = OneClassSVM(gamma='auto', **ocsvm_params).fit(codes)
    outliers = ocsvm.predict(codes) == -1
    return codes[outliers], np.nonzero(outliers)[0], ocsvm


def isolation_forrest(codes, **if_params):
    isof = IsolationForest(**if_params).fit(codes)
    outliers = isof.predict(codes) == -1
    return codes[outliers], np.nonzero(outliers)[0], isof