import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


### 사용자 정의 함수 선언
def make_binary_matrix(matrix, threshold):
    # 임계값 이하의 원소들을 0으로 설정
    binary_matrix = matrix.apply(lambda x: np.where(x > threshold, 1, 0))
    return binary_matrix

def filter_matrix(matrix, threshold):
    # 임계값 이하의 원소들을 0으로 설정
    filtered_matrix = matrix.where(matrix > threshold, 0)
    return filtered_matrix

# 임계 값을 0-1까지로, 25%로 x축을 한정해서 시각화, 최대 변화율 지점의 x축 값 찾기
@st.cache_data 
def threshold_count(matrix):
    L = matrix
    element_counts = []

    # 임계값 생성
    threshold_values = np.linspace(0, 1, 1000)[:250]

    # 각 임계값에 대해 생존값 계산
    for threshold in threshold_values:
        thresholded_matrix = filter_matrix(L, threshold)
        thresholded_matrix = thresholded_matrix.copy().to_numpy()

        np.fill_diagonal(thresholded_matrix, 0)  # 대각선 원소는 0으로 설정
        count = (thresholded_matrix >= threshold).sum().sum()
        element_counts.append(count)

    # 최대 변화율(절대값) 찾기
    max_change = 0
    max_change_index = 0
    for i in range(1, len(element_counts)):
        change = abs(element_counts[i - 1] - element_counts[i])
        if change > max_change:
            max_change = change
            max_change_index = i
    
    # df_graph = pd.DataFrame({'x': threshold_values, 'y': element_counts})
    max_change_threshold = threshold_values[max_change_index]

    # 그래프 그리기
    fig, ax = plt.subplots() # 수정된 부분
    ax.plot(threshold_values, element_counts)
    ax.set_xlabel('Threshold Value') # ax를 사용하여 라벨 설정
    ax.set_ylabel('Number of Elements >= Threshold')
    ax.set_title('Number of Elements Greater than or Equal to Threshold in a Matrix')

    # 최대 변화율 지점 표시
    ax.plot(max_change_threshold, element_counts[max_change_index], 'ro') # ax를 사용하여 데이터 표시

    ax.grid(True)
    st.pyplot(fig) # 수정된 부분
    st.write(f'생존율 급감 구간의 임계 값 : {max_change_threshold}')

    return plt.show()

@st.cache_data()
def get_submatrix_withlabel(df, start_row, start_col, end_row, end_col, first_index_of_df, numberoflabel = 2):
    row_indexs = list(range(first_index_of_df[0]-numberoflabel, first_index_of_df[0])) + list(range(start_row, end_row+1))
    col_indexs = list(range(first_index_of_df[1]-numberoflabel, first_index_of_df[1])) + list(range(start_col, end_col+1))
    # print(row_indexs)
    # print(col_indexs)

    submatrix_withlabel = df.iloc[row_indexs, col_indexs]
    return submatrix_withlabel

def reduce_negative_values(df, first_idx, mid_ID_idx):
    # 데이터프레임 복사
    df_editing = df.copy()

    # 숫자형 데이터로 변환 (first_idx[0] 행부터, first_idx[1] 열부터 끝까지)
    df_test = df_editing.iloc[first_idx[0]:mid_ID_idx[0], first_idx[1]:mid_ID_idx[1]].apply(pd.to_numeric, errors='coerce')

    # 음수 값이 있는 위치 추적 및 줄인 값 계산
    reduced_values_per_column = {}

    def reduce_and_track(value, col_index):
        if value < 0:
            # 줄일 값 저장 (음수 값의 절반)
            reduced_value = value / 2
            if col_index not in reduced_values_per_column:
                reduced_values_per_column[col_index] = 0
            reduced_values_per_column[col_index] += value - reduced_value  # 원래 값 - 절반으로 줄인 값
            return reduced_value
        return value

    # 음수인 값만 1/2로 줄이면서 추적
    for col_idx in range(df_test.shape[1]):
        df_test.iloc[:, col_idx] = df_test.iloc[:, col_idx].apply(lambda x: reduce_and_track(x, col_idx))

    # 수정된 값을 원본 데이터프레임에 다시 반영
    df_editing.iloc[first_idx[0]:, first_idx[1]:] = df_test

    # 마지막 행에 줄인 값만큼 더하기
    last_row_index = df_editing.shape[0] - 1
    for col_idx, total_reduced in reduced_values_per_column.items():
        df_editing.iloc[last_row_index, first_idx[1] + col_idx] -= total_reduced

    msg = "음수 값들을 절반으로 줄이고, 줄인 값을 마지막 행에 더했습니다."

    # 중간 인덱스 값은 그대로 반환 (mid_ID_idx는 행과 열 인덱스이므로 이 경우 변경되지 않음)
    return df_editing, msg, mid_ID_idx



def get_mid_ID_idx(df, first_idx):
    matrix_X = df.iloc[first_idx[0]:, first_idx[1]:].astype(float)
    row_cnt, col_cnt, row_sum, col_sum = 0, 0, 0, 0
    for v in matrix_X.iloc[0]:
        if abs(row_sum - v) < 0.001:
            if v == 0:
                continue
            else: break
        row_cnt += 1
        row_sum += v
    for v in matrix_X.iloc[:, 0]:
        print(f'gap: {col_sum-v}, sum: {col_sum}, value: {v}')
        if abs(col_sum - v) < 0.001:
            if v == 0:
                continue
            else: break
        col_cnt += 1
        col_sum += v
    
    if row_cnt == col_cnt:
        size = row_cnt
    else:
        size = max(row_cnt, col_cnt)

    return (first_idx[0]+size, first_idx[1]+size)

def insert_row_and_col(df, first_idx, mid_ID_idx, code, name, num_of_label):
    df_editing = df.copy()
    df_editing.insert(loc=mid_ID_idx[1], column='a', value=np.nan, allow_duplicates=True)
    df_editing.iloc[first_idx[0]-num_of_label, mid_ID_idx[1]] = code
    df_editing.iloc[first_idx[0]-num_of_label+1, mid_ID_idx[1]] = name
    df_editing.iloc[first_idx[0]:, mid_ID_idx[1]] = 0
    df_editing.columns = range(df_editing.shape[1])
    df_editing = df_editing.T   
    df_editing.insert(loc=mid_ID_idx[0], column='a', value=np.nan, allow_duplicates=True)
    df_editing.iloc[first_idx[1]-num_of_label, mid_ID_idx[0]] = code
    df_editing.iloc[first_idx[1]-num_of_label+1, mid_ID_idx[0]] = name
    df_editing.iloc[first_idx[1]:, mid_ID_idx[0]] = 0
    df_editing.columns = range(df_editing.shape[1])
    df_editing = df_editing.T
    df_inserted = df_editing.copy()
    mid_ID_idx = (mid_ID_idx[0]+1, mid_ID_idx[1]+1)
    msg = f'A new row and column (Code: {code}, Name: {name}) have been inserted.'

    return df_inserted, mid_ID_idx, msg

def transfer_to_new_sector(df, first_idx, origin_code, target_code, ratio, code_label = 2):
    df_editing = df.copy()
    target_idx = df_editing.index[df_editing[first_idx[1]-code_label] == target_code].tolist()
    if len(target_idx) == 1:
        target_idx = target_idx[0]
    else:
        msg = 'ERROR: target code is not unique.'
        return df_editing, msg
    origin_idx = df_editing.index[df_editing[first_idx[1]-code_label] == origin_code].tolist()
    if len(origin_idx) == 1:
        origin_idx = origin_idx[0]
    else:
        msg = 'ERROR: origin code is not unique.'
        return df_editing, msg
    df_editing.iloc[first_idx[0]:, first_idx[1]:] = df_editing.iloc[first_idx[0]:, first_idx[1]:].apply(pd.to_numeric, errors='coerce')
    origin_idx = (origin_idx, origin_idx-first_idx[0]+first_idx[1])
    target_idx = (target_idx, target_idx-first_idx[0]+first_idx[1])
    df_editing.iloc[target_idx[0] ,first_idx[1]:] += df_editing.iloc[origin_idx[0] ,first_idx[1]:] * ratio
    df_editing.iloc[origin_idx[0] ,first_idx[1]:] = df_editing.iloc[origin_idx[0] ,first_idx[1]:] * (1-ratio)
    df_editing.iloc[first_idx[0]: ,target_idx[1]] += df_editing.iloc[first_idx[0]: ,origin_idx[1]] * ratio
    df_editing.iloc[first_idx[0]: ,origin_idx[1]] = df_editing.iloc[first_idx[0]: ,origin_idx[1]] * (1-ratio)

    msg = f'{ratio*100}% of {origin_code} has been moved to {target_code}.'
    return df_editing, msg

def remove_zero_series(df, first_idx, mid_ID_idx):
    df_editing = df.copy()
    df_test = df_editing.copy()
    df_test = df_editing.iloc[first_idx[0]:, first_idx[1]:].apply(pd.to_numeric, errors='coerce')
    zero_row_indices = df_test.index[(df_test == 0).all(axis=1)].tolist()
    zero_row_indices = [item for item in zero_row_indices if item>=first_idx[0] and item<=mid_ID_idx[0]]
    zero_col_indices = list(map(lambda x: x - first_idx[0] + first_idx[1], zero_row_indices))
    df_editing.drop(zero_row_indices, inplace=True)
    df_editing.drop(zero_col_indices, inplace=True, axis=1)
    df_editing.columns = range(df_editing.shape[1])
    df_editing.index = range(df_editing.shape[0])
    count = len(zero_col_indices)
    msg = f'{count}개의 행(열)이 삭제되었습니다.'
    mid_ID_idx = (mid_ID_idx[0] - count, mid_ID_idx[1] - count)
    return df_editing, msg, mid_ID_idx

def donwload_data(df, file_name):
    csv = convert_df(df)
    button = st.download_button(label=f"{file_name} 다운로드", data=csv, file_name=file_name, mime='text/csv')
    return button


@st.cache_data()
def load_data(file):
    st.session_state['df'] = pd.read_excel(file, header=None)
    return st.session_state['df']

@st.cache_data 
def convert_df(df):
    return df.to_csv(header=False, index=False).encode('utf-8-sig')
