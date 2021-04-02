
import numpy as np
from scipy.stats import mode

def scaling(images,method='mean'):
    '''
    scale each image to have either a mean, median, or mode of 1


    params:
        images - 3D array with num_images x num_rows x num_columns
        method - method of scaling 'mean', 'median', or 'mode'
    '''
    num_images=np.shape(images)[0]
    num_rows=np.shape(images)[1]
    num_columns=np.shape(images)[2]

    scaled_images=np.zeros(shape=(num_images,num_rows,num_columns))

    for i in range(num_images):

        # Find central value
        if method=='mean':
            scale_value=np.nanmean(images[i,:,:].flatten())
        elif method=='median':
            scale_value=np.nanmedian(images[i,:,:].flatten())
        elif method=='mode':
            scale_value=mode(images[i,:,:].flatten(),nan_policy='omit')

        # Scale image to central value of 1
        scaled_images[i,:,:]=images[i,:,:]/scale_value

    return scaled_images

def sigma_clip(scaled_images,N=3,method='mean',max_iter=3):
    '''
    for each pixel, iteratively clip values that are more than N sigma away from the mean or median value for that pixel across all images

    params:
        scaled_images
        N - sigma multiplier for clipping
        method - method for estimating the central value for sigma clipping, 'mean', or 'median'
        max_iter - maximum number of passes through the images for convergence
    returns:
        clipped_images
        mask - same shape as clipped_images, is False if this location is clipped, is True if not clipped
    '''
    num_images=np.shape(scaled_images)[0]
    num_rows=np.shape(scaled_images)[1]
    num_columns=np.shape(scaled_images)[2]

    clipped_images=scaled_images
    mask=np.array(np.ones(shape=(num_images,num_rows,num_columns)),dtype=bool)

    converged=False
    count=0
    while not converged:
        count+=1
        print('Starting Sigma-Clip Iteration '+str(count))

        new_clip=False # has the clipping algorithm clipped a point this pass?

        for i in range(num_rows):
            for j in range(num_columns):

                pixel_values=clipped_images[:,i,j] # values at this pixel across all images

                if method=='mean':
                    central_value=np.mean(pixel_values[mask[:,i,j]])
                elif method=='median':
                    central_value=np.median(pixel_values[mask[:,i,j]])

                sigma=np.std(pixel_values[mask[:,i,j]])
                distance=np.absolute(pixel_values-central_value) # distance from central value at each image

                mask_clip=np.where(distance>sigma*N,True,False) # True if this index needs to be clipped

                #if np.count_nonzero(mask_clip)>0:
                    #print(mask_clip)

                mask_clip_new=np.logical_and(mask_clip,mask[:,i,j]) # true if this index needs to be clipped and is not already clipped
                if np.count_nonzero(mask_clip_new)>0:
                    #print('Clipped {:.0f} images of pixel {:.0f},{:.0f}'.format(np.count_nonzero(mask_clip_new),i,j))

                    new_clip=True
                    mask[mask_clip_new,i,j]=False
                    clipped_images[mask_clip_new,i,j]=0


        if ((not new_clip) or (count>=max_iter)):
            # If the clipping algorithm has not clipped any points this pass, it has converged
            converged=True

    num_keep=np.sum(mask)
    num_total=num_images*num_rows*num_columns
    num_clip=num_total-num_keep
    print('Clipped {:.1f} percent of pixels'.format(100*num_clip/num_total))

    return clipped_images,mask



def weighted_mean(raw_images,mask=None,weight_method='mean',combine_method='median'):
    '''
    Takes a weighted mean at each pixel over all images
    NOTE: When a pixel gets sigma clipped, you must remove its weight from the denominator of the sum
    weight values can be the mean/median value of the original image
    W=sum{w_i x_i}/sum{w_i}

    params:
        raw_images - num_images,num_rows,num_columns flat field images overscan subtracted and trimmed but not scaled
        mask - same shape as clipped_images, True if pixel is not clipped, false if clipped
               if mask=None all pixel values will be included
        weight_method - method for determining the weights of each image in clipped_images, 'mean', 'median', or 'mode'
        combine_method - method for combining images into a master image, 'mean' or 'median'
    '''
    num_images=raw_images.shape[0]
    num_rows=raw_images.shape[1]
    num_columns=raw_images.shape[2]
    
    if mask is None:
        # all true values, same shape as raw_images
        mask=np.array(np.ones(shape=(num_images,num_rows,num_columns)),dtype=bool)
        
    scaled_images=scaling(raw_images)

    # Set raw_images values to NAN where mask is False
    # where mask=True, return value from raw_images
    # where mask=False, return np.nan
    # masked_images will be used to determine weights, so it is not scaled
    masked_images=np.where(mask,raw_images,np.nan)
    # masked_images_scaled will be used for the actual weighted average
    masked_images_scaled=np.where(mask,scaled_images,np.nan)
    
    # Find weight values
    if weight_method=='mean':
        weights=np.nanmean(masked_images,axis=(1,2))
    elif weight_method=='median':
        weights=np.nanmedian(masked_images,axis=(1,2))
        
    #normalize weights to 1 (this shouldn't be strictly necessary)
    scale_value=np.mean(weights)
    weights_norm=weights/scale_value

    # for np.ma.array, True indicates a masked point (this is the opposite of the system I've created oops)
    # inverting the mask to correct for this
    masked_array=np.ma.array(masked_images_scaled,mask=np.invert(mask))

    # Take the weighted average over all images
    if combine_method=='mean':
        weighted_image=np.ma.average(masked_array,axis=0,weights=weights_norm)
    elif combine_method=='median':
        # can't really have a weighted median 
        weighted_image=np.ma.median(masked_array,axis=0)

    return weighted_image





