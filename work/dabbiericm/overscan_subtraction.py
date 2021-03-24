import numpy as np
from statsmodels.regression.linear_model import OLS

def overscan_subtraction(image,overscan,trim,combine_method,fit_type,fit_deg):
    '''
    given an image with an overscan region, performs an overscan subtraction with a row-wise polynomial fit of the overscan region

    params:
        image - 2D array with image (including overscan region)
        overscan - list with [min_column,max_column] of overscan region (indexed from 0)
        trim - list with [min_column,max_column] of image region (indexed from 0)
        combine_method - method for collapsing columns of overscan region, "median" or "mean"
        fit_type - method for row-wise fit of overscan region "chebychev", "legendre", "hermite", or "polynomial"
        fit_deg - degree of row-wise fit of overscan region
    if fit_type is polynomial returns:
        image_sub_trim - trimmed and overscan subtracted image
        AIC - AIC for polynomial fit across rows
    otherwise returns:
        image_sub_trim - trimmed and overscan subtracted image
    '''

    # collect overscan portion of the image
    image_overscan=image[:,overscan[0]:overscan[1]]

    # collapse the columns of the overscan using median or mean
    if combine_method=='mean':
        combine_overscan=np.mean(image_overscan,axis=1)
    elif combine_method=='median':
        combine_overscan=np.median(image_overscan,axis=1)
    else:
        raise ValueError('combine_method must be mean or median')

    rows=np.arange(len(combine_overscan))

    if fit_type=='polynomial':
        # perform a polynomial fit across the rows of the combined overscan
        p=np.flip(np.polyfit(x=rows,y=combine_overscan,deg=fit_deg)) #flipped so 0 order term is first
        beta=p.reshape((len(p),1))
        # Now create design matrix
        X=np.zeros((len(rows),fit_deg+1))
        for column in range(X.shape[1]):
            X[:,column]=rows**column
        # do some matrix multiplication to get the fit
        fit=np.matmul(X,beta).flatten() # flattened so it's not a column matrix

        # AIC calculation to polynomial fit for model selection over fit degree
        regr = OLS(combine_overscan, X).fit()
        AIC=regr.aic
    elif fit_type=='chebychev':
        p=np.polynomial.chebyshev.Chebyshev.fit(x=rows,y=combine_overscan,deg=fit_deg).convert().coef
        fit=np.polynomial.chebyshev.chebval(x=rows,c=p)
    elif fit_type=='legendre':
        p=np.polynomial.legendre.Legendre.fit(x=rows,y=combine_overscan,deg=fit_deg).convert().coef
        fit=np.polynomial.legendre.legval(x=rows,c=p)
    elif fit_type=='hermite':
        p=np.polynomial.hermite.Hermite.fit(x=rows,y=combine_overscan,deg=fit_deg).convert().coef
        fit=np.polynomial.hermite.hermval(x=rows,c=p)
    else:
        raise ValueError('fit_type must be polynomial, chebychev, legandre, or hermite')

    image_subtract=np.zeros((image.shape[0],image.shape[1]))
    # now subtract out fit row-wise
    for row in range(len(rows)):
        image_subtract[row,:]=image[row,:]-fit[row]

    # finally trim off overscan
    image_sub_trim=image_subtract[:,trim[0]:trim[1]]

    if fit_type=='polynomial':
        return image_sub_trim,AIC
    return image_sub_trim
