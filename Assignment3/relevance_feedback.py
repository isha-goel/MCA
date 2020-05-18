def relevance_feedback(vec_docs, vec_queries, sim, gt, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """

    for i in range(0,30):
      r = list()
      nr = list()
      ranked_documents = np.argsort(-sim[:, i])
      top_N = ranked_documents[:10]

      for j in gt:
        if(j[0] == i+1):
          if(j[1] - 1 in top_N):
            r.append(j[1] -1)

      for j in top_N:
        if j not in r:
          nr.append(j)

      sum_r = 0
      for j in r:
        sum_r += vec_docs[j]

      sum_nr = 0
      for j in nr:
        sum_nr += vec_docs[j]

      vec_queries[i] = vec_queries[i] + 0.7*sum_r - 0.3*sum_nr
      
    rf_sim = cosine_similarity(vec_docs, vec_queries)
    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, gt, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    for i in range(0,30):
      top_relevant = list()
      non_relevant = list()
      relevant_gt = list()
      ranked_documents = np.argsort(-sim[:, i])
      top_N = ranked_documents[:10]
      
      for j in gt:
        if(j[0] == i+1):
          relevant_gt.append(j[1] -1)

      for j in gt:
        if(j[0] == i+1):
          if(j[1] - 1 in top_N):
            top_relevant.append(j[1] -1)

      for j in top_N:
        if j not in top_relevant:
          non_relevant.append(j)

      sum_r = 0
      for j in top_relevant:
        sum_r += vec_docs[j]

      sum_nr = 0
      for j in non_relevant:
        sum_nr += vec_docs[j]

      vec_queries[i] = vec_queries[i] + 0.7*sum_r - 0.3*sum_nr
      
      docs = vec_docs.toarray()

      top = list()
      for j in relevant_gt:
        d = docs[j].flatten()
        idx = np.argsort(d)
        idx = idx[-10:]
        for k in idx:
          top.append((d[k], k))

      top = sorted(top, key = lambda x: x[0])
      top_k = top[-10:]

      for j in top_k:
        a = np.zeros((1,10625))
        a[0,j[1]] = j[0]
        vec_queries[i] = vec_queries[i] + a
        
    rf_sim = cosine_similarity(vec_docs, vec_queries)
    return rf_sim
