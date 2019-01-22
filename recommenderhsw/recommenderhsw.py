import numpy as np
import pandas as pd
from scipy import spatial

class Recommender:
    
    '''
    Recommender(sample_df, similarity_func, target, closer_count)
    similarity_func: "euclidean" or "cosine"
    target: "user_id"
    '''
    
    def __init__(self, sample_df, similarity_func, target, closer_count):
        self.sample_df = sample_df
        self.similarity_func = similarity_func
        self.target = target
        self.closer_count = closer_count
        import numpy as np
        import pandas as pd
        from scipy import spatial
    
    def _delete_zero(self, vector_1, vector_2):
        
        idx = np.array(vector_1).nonzero()[0]
        vector_1 = np.array(vector_1)[idx]
        vector_2 = np.array(vector_2)[idx]

        idx = np.array(vector_2).nonzero()[0]
        vector_1 = np.array(vector_1)[idx]
        vector_2 = np.array(vector_2)[idx]
        
        return vector_1, vector_2, idx
    
    
    def _euclidean_similarity(self, vector_1, vector_2):
        '''
	euclidean_similarity
        '''
        
        vector_1, vector_2, _ = self._delete_zero(vector_1, vector_2)
        
        return np.linalg.norm(vector_1 - vector_2)
    
    def _cosine_similarity(self, vector_1, vector_2):
        '''
        cosine_similarity
	'''

        vector_1, vector_2, _ = self._delete_zero(vector_1, vector_2)
        
        return 1 - spatial.distance.cosine(vector_1, vector_2)
    
    def similarity_matrix(self):
    
        # index 데이터 저장
        index = self.sample_df.index

        matrix = []
        for idx_1, value_1 in self.sample_df.iterrows():
            row = []
            for idx_2, value_2 in self.sample_df.iterrows():
                if self.similarity_func == "euclidean":
                    row.append(self._euclidean_similarity(value_1, value_2))
                if self.similarity_func == "cosine":
                    row.append(self._cosine_similarity(value_1, value_2))
                    
            matrix.append(row)   
        
        self.sm_df = pd.DataFrame(matrix, columns=index, index=index)
        
        return self.sm_df
    
    def mean_score(self):
        
        self.similarity_matrix()
        
        ms_df = self.sm_df.drop(self.target)
        ms_df = ms_df.sort_values(self.target, ascending=False)
        ms_df = ms_df[:self.closer_count]
        ms_df = self.sample_df.loc[ms_df.index]

        # pred_df 결과 생성
        pred_df = pd.DataFrame(columns=self.sample_df.columns)
        pred_df.loc["user"] = self.sample_df.loc[self.target]
        pred_df.loc["mean"] = ms_df.mean()
        
        self.pred_df = pred_df
        
        return self.pred_df
    
    def recommend(self):
        self.mean_score()
        
        recommend_df = self.pred_df.T
        recommend_df = recommend_df[recommend_df["user"] == 0]
        recommend_df = recommend_df.sort_values("mean", ascending=False)

        self.recommend_df = list(recommend_df.index)
        
        return self.recommend_df
           
    # MSE
    def _mse(self, value, pred):

        value, pred, idx = self._delete_zero(value, pred)

        return sum((value - pred) ** 2) / len(idx)
    
    # RMSE
    def _rmse(self, value, pred):

        value, pred, idx = self._delete_zero(value, pred)

        return np.sqrt(sum((value - pred) ** 2) / len(idx))
    
    # MAE
    def _mae(self, value, pred):

        vector_1, vector_2, idx = self._delete_zero(value, pred)

        return sum(abs(value - pred)) / len(idx)
    
    # 전체 추천 모델에 대한 성능 평가
    def evaluate(self, algorithm):
        '''
        algorithm: "mse", "rmse", "mae"
        '''
        
        self.mean_score()
        
        # user 리스트
        users = self.sample_df.index

        # user 별 envaluate값의 모음
        evaluate_list = []

        for target in users:
            # 하나의 user에 대한 예측 값을 얻음
            if algorithm == "mse":
                evaluate_var = self._mse(self.pred_df.loc["user"], self.pred_df.loc["mean"])
            elif algorithm == "rmse":
                evaluate_var = self._rmse(self.pred_df.loc["user"], self.pred_df.loc["mean"])
            elif algorithm == "mae":
                evaluate_var = self._mae(self.pred_df.loc["user"], self.pred_df.loc["mean"])
            
            evaluate_list.append(evaluate_var)

        return np.average(evaluate_list)
