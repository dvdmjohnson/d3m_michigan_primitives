import numpy as np

def get_random_constraints(labels, numML, numCL, randomState):
    """ Utility method for generating pairwise constraints from labels.
        Warning: if more constraints are requested than can be uniquely
        generated for one or more classes, method will run forever.

    Arguments:
        labels: an n-lenth vector containing the class labels for each instance.
            Should be normalized so that the set of label values ranges from
            0 to c-1, where c is the number of classes
            numML: the number of must-link constraints per class to sample
            numCL: the number of cannot-link constraints per class to sample
            randomState: a numpy RandomState object to be used by this method call

    Returns:
        cons: the list of pairwise constraints on the data - an m by 3 numpy
            array, with each row containing a constraint in the form 
            <id1, id2, value>,where id1 and id2 are indexes of instances in
            the training data and value is either 0 (indicating a must-link
            constraint) or 1 (indicating a cannot-link constraint).

    Raises:
        AssertError: if there are not at least two classes in the input label set,
            or the labels are not normalized properly
    """
        
    # initialize
    labset = np.unique(labels)
    c = len(labset)
    assert c >= 2
    assert labset[0] == 0
    assert labset[c - 1] == c - 1
    cs = np.zeros(((numCL + numML) * c, 3), dtype=np.int32)
    conset = set() 
    
    # iterate through classes
    for i in range(c):
        # establish class membership lists
        thisclass = np.nonzero(labels == labset[i])[0]
        otherclass = np.nonzero(labels != labset[i])[0]
        
        # get must-link constraints
        c1 = randomState.randint(len(thisclass), size=numML)
        c2 = randomState.randint(len(thisclass), size=numML)
        
        # ensure no instance is constrained with itself and all constraints are unique
        for j in range(numML):
            while (c1[j] == c2[j] or (thisclass[c1[j]], thisclass[c2[j]]) in conset):
                c1[j] = randomState.randint(len(thisclass))
                c2[j] = randomState.randint(len(thisclass))
            # if valid, add constraint to constraint set
            conset.add((thisclass[c1[j]], thisclass[c2[j]]))
            conset.add((thisclass[c2[j]], thisclass[c1[j]]))
        
        # add constraints to output array
        c3 = np.zeros(numML)
        cs[i * (numML + numCL):i * (numML + numCL) + numML] = np.vstack((thisclass[c1], thisclass[c2], c3)).T
        
        # get cannot-link constraints
        c1 = randomState.randint(len(thisclass), size=numCL)
        c2 = randomState.randint(len(otherclass), size=numCL)
        # ensure uniqueness
        for j in range(numCL):
            while ((thisclass[c1[j]], otherclass[c2[j]]) in conset):
                c1[j] = randomState.randint(len(thisclass))
                c2[j] = randomState.randint(len(otherclass))
            # if valid, add constraint to constraint set
            conset.add((thisclass[c1[j]], otherclass[c2[j]]))
            conset.add((otherclass[c2[j]], thisclass[c1[j]]))
        
        # add constraints to output array
        c3 = np.ones(numCL)
        cs[i * (numML + numCL) + numML:(i + 1) * (numML + numCL)] = np.vstack((thisclass[c1], otherclass[c2], c3)).T
    return cs

def normalize_labels(labs):
    """ Normalizes a set of input labels to a 1-dimensional ndarray of int32s,
        with labels ranging from 0 to c-1, where c is the number of different
        labels.
        
    Arguments:
        labs: a 1-dimensional array-like of labels (will be flattened if not 1-dimensional)
        
    Returns:
        nlabs: a normalized copy of labs
        
    Raises:
        None
    """
    labs = np.int32(np.array(labs)).flatten()
    labset = np.unique(labs)
    newlabs = np.arange(len(labset))
    nlabs = np.zeros_like(labs)
    for i in range(len(labset)):
        nlabs[labs == labset[i]] = newlabs[i]
    return nlabs
