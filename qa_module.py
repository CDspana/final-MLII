# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 00:25:55 2024

@author: user
"""

from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
#import torch
import numpy as np


class QA():
  def __init__(self, file_path):
    self.file_path = file_path
    self.tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
    self.model = BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')

    # Precalcular los embeddings al inicializar la clase
    self.chunks_embeddings = self.calculate_chunks_embeddings()

  def read_text_file(self):
      with open(self.file_path, 'r', encoding='utf-8') as file:
          text = file.read()
      text = text.replace('\n', ' ')
      return text

  def split_into_chunks(self, text):
      chunks = text.split('.')
      chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
      return chunks

  def calculate_chunks_embeddings(self):
        text = self.read_text_file()
        chunks = self.split_into_chunks(text)

        chunks_embeddings = {}
        for chunk in chunks:
            embeddings = self.generate_embeddings(chunk)
            # Squeeze to ensure the embeddings are 2D
            chunks_embeddings[chunk] = np.squeeze(embeddings)

        return chunks_embeddings

  def generate_embeddings(self, text):
      inputs = self.tokenizer(text, return_tensors='pt')
      outputs = self.model(**inputs)
      embeddings = outputs.last_hidden_state.mean(dim=1)
      return embeddings.detach().numpy()

  def get_responses(self, question, n=2):
      question_embedding = self.generate_embeddings(question)

      relevance_scores = []
      for chunk, chunk_embedding in self.chunks_embeddings.items():
          # Ensure the question_embedding and chunk_embedding are 2D arrays
          similarity_score = cosine_similarity([np.squeeze(question_embedding)], [chunk_embedding])[0][0]
          relevance_scores.append((chunk, similarity_score))

      sorted_chunks = sorted(relevance_scores, key=lambda x: x[1], reverse=True)[:n]
      return sorted_chunks
