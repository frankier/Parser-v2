#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Timothy Dozat
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.





import sys
import numpy as np
import tensorflow as tf

from nparser import Configurable
from nparser import Bucket
from nparser.misc.colors import ctext

#***************************************************************
class Multibucket(Configurable):
  """ """
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """ """
    
    self._embed_model = kwargs.pop('embed_model', None)
    super(Multibucket, self).__init__(*args, **kwargs)
    
    self._indices = []
    self._buckets = []
    self._len2idx = {}
    self.placeholder = None
    self._embed_models = [] # place to keep embedding models to prevent reinitializing with every batch
    return
  
  #=============================================================
  def __call__(self, vocab, keep_prob=None, moving_params=None):
    """ """
    
    # This placeholder is used to ensure the bucket data is in the right order
    reuse = None if moving_params is None else True
    self.generate_placeholder()
    embeddings = []
    for i, bucket in enumerate(self):
      if i > 0:
        reuse = True
      with tf.variable_scope(self.name+'-multibucket', reuse=reuse):
        embeddings.append(bucket(vocab, keep_prob=keep_prob, moving_params=moving_params))
    return tf.nn.embedding_lookup(tf.concat(embeddings, axis=0), self.placeholder)
  
  #=============================================================
  def reset_placeholders(self):
    self.placeholder = None
    for bucket in self:
      bucket.reset_placeholders()
    return

  #=============================================================
  def generate_placeholder(self):
    """ """
    
    if self.placeholder is None:
      self.placeholder = tf.placeholder(tf.int32, shape=(None,), name=self.name+'-multibucket')
    return self.placeholder
  
  #=============================================================
  def open(self, maxlens, depth=None):
    """ """
    
    # create rnn embedding models if needed
    if len(self._embed_models)==0:
      for idx, maxlen in enumerate(maxlens): # i.e. how many buckets there is
        if self.embed_model != None:
          self._embed_models.append(self.embed_model.from_configurable(self, name='%s-%d' % (self.name, idx))) # initialize embedding model, name is the bucket name
        else:
          self._embed_models.append(None)

    self._indices = [(0,0)]
    self._buckets = []
    self._len2idx = {}
    prevlen = -1
    for (idx, maxlen), bucket_embed_model in zip(enumerate(maxlens), self._embed_models):
      self._buckets.append(Bucket.from_configurable(self, embed_model=bucket_embed_model, name='%s-%d' % (self.name, idx)).open(maxlen, depth=depth)) # use the same rnn embedding model instead of creating a new one
      self._len2idx.update(list(zip(list(range(prevlen+1, maxlen+1)), [idx]*(maxlen-prevlen))))
      prevlen = maxlen
    return self
  
  #=============================================================
  def add(self, idxs, tokens=None):
    """ """
    
    if isinstance(self.indices, np.ndarray):
      raise TypeError("The buckets have already been closed, you can't add to them")
    
    bkt_idx = self._len2idx.get(len(idxs), len(self)-1)
    idx = self[bkt_idx].add(idxs, tokens=tokens)
    self.indices.append( (bkt_idx, idx) )
    return len(self.indices) - 1

  def extend_closed(self, idxs_batch):
    if not isinstance(self.indices, np.ndarray):
      raise TypeError("The buckets have not yet been closed, you can't extend_closed them")
    extensions = [[] for _ in self]
    own_new_indices = []
    for idxs in idxs_batch:
      bkt_idx = self._len2idx.get(len(idxs), len(self)-1)
      extensions[bkt_idx].append(idxs)
      own_new_indices.append((bkt_idx, len(self[bkt_idx].indices)))
    old_indices_len = len(self.indices)
    new_indices_len = old_indices_len + len(own_new_indices)
    self.indices.resize(new_indices_len)
    self.indices[old_indices_len:] = own_new_indices
    for idx, all_idxs in enumerate(extensions):
      bkt_indices_shape = self[idx].indices.shape
      all_idxs_len = len(all_idxs)
      new_len = bkt_indices_shape[0] + all_idxs_len
      self[idx].indices.resize((new_len, *bkt_indices_shape[1:]), refcheck=False)
      for offset, idxs in zip(range(all_idxs_len, new_len), all_idxs):
        self[idx].indices[offset,0:len(idxs)] = idxs
    return range(old_indices_len, new_indices_len)
  
  #=============================================================
  def close(self):
    """ """
    
    for bucket in self:
      bucket.close()
    
    self._indices = np.array(self.indices, dtype=[('bkt_idx', 'i4'), ('idx', 'i4')])
    return
  
  #=============================================================
  def inv_idxs(self):
    """ """
    
    return np.argsort(np.concatenate([np.where(self.indices['bkt_idx'][1:] == bkt_idx)[0] for bkt_idx in range(len(self))]))
  
  #=============================================================
  def get_tokens(self, bkt_idx, batch):
    """ """

    return self[bkt_idx].get_tokens(batch)

  #=============================================================
  @classmethod
  def from_dataset(cls, dataset, *args, **kwargs):
    """ """
    
    multibucket = cls.from_configurable(dataset, *args, **kwargs)
    indices = []
    for multibucket_ in dataset:
      indices.append(multibucket_.indices)
    #for i in xrange(1, len(indices)):
    #  assert np.equal(indices[0].astype(int), indices[i].astype(int)).all()
    multibucket._indices = np.array(multibucket_.indices)
    buckets = [Bucket.from_dataset(dataset, i, *args, **kwargs) for i in range(len(multibucket_))]
    multibucket._buckets = buckets
    if dataset.verbose:
      for bucket in multibucket:
        print('Bucket {name} is {shape}'.format(name=bucket.name, shape=ctext(' x '.join(str(x) for x in bucket.indices.shape), 'bright_blue')),file=sys.stderr)
    return multibucket
  
  #=============================================================
  @property
  def indices(self):
    return self._indices
  @property
  def embed_model(self):
    return self._embed_model
  
  #=============================================================
  def __str__(self):
    return str(self._buckets)
  def __iter__(self):
    return (bucket for bucket in self._buckets)
  def __getitem__(self, key):
    return self._buckets[key]
  def __len__(self):
    return len(self._buckets)
  def __enter__(self):
    return self
  def __exit__(self, exception_type, exception_value, trace):
    if exception_type is not None:
      raise exception_type(exception_value)
    self.close()
    return
