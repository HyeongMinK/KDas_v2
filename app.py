import numpy as np
import pandas as pd
import streamlit as st
from functions import *


### Streamlit 구현
def main():
    st.sidebar.header("다운로드")
    st.title("DasHboard beta 1.2")
    mode = st.radio('모드 선택', ['Korea(2010~2020)', 'Korea(1990~2005)', 'Manual'])
    if mode == 'Korea(2010~2020)':
        first_idx = (6,2)
        subplus_edit =False
        number_of_label = 2
    elif mode == 'Korea(1990~2005)':
        first_idx = (5,2)
        subplus_edit =True
        number_of_label = 2
    else:
        first_idx = 0
        subplus_edit =False
        number_of_label = 2

    if 'number_of_divide' not in st.session_state:
        st.session_state['number_of_divide'] = 0
    def find_string_values(df, first_idx):
        # 특정 구간의 데이터 선택
        selected_df = df.iloc[first_idx[0]:, first_idx[1]:]

        # 문자열이 포함된 셀의 위치를 저장할 리스트
        string_locations = []

        # 모든 셀을 순회하며 문자열이 있는지 확인
        for row_idx, row in selected_df.iterrows():
            for col_idx, value in row.items():
                if isinstance(value, str):  # 문자열인지 확인
                    string_locations.append((row_idx, col_idx, value))

        return string_locations
    # 문자열이 포함된 위치를 NA로 대체하는 함수
    def replace_string_with_na(df, string_locations):
        for row_idx, col_idx, _ in string_locations:
            df.iloc[row_idx, col_idx] = np.nan  # 해당 위치의 값을 pd.NA로 대체

    def slice_until_first_non_nan_row(df):
        # DataFrame의 맨 아래부터 위로 순회하며 NaN이 아닌 첫 번째 행 찾기
        last_valid_index = None
        for row_idx in reversed(range(df.shape[0])):  # 아래에서 위로 순회
            if not df.iloc[row_idx].isna().all():  # NaN이 아닌 행을 찾으면
                last_valid_index = row_idx
                break

        # NaN이 아닌 행까지 슬라이싱 (찾지 못한 경우 전체 슬라이스)
        if last_valid_index is not None:
            sliced_df = df.iloc[:last_valid_index + 1]
        else:
            sliced_df = pd.DataFrame()  # 모든 행이 NaN인 경우 빈 DataFrame 반환

        return sliced_df, last_valid_index

    # 파일 업로드 섹션s
    st.session_state['uploaded_file'] = st.file_uploader("여기에 파일을 드래그하거나 클릭하여 업로드하세요.", type=['xls', 'xlsx'])
    if 'df' not in st.session_state:
        if st.session_state['uploaded_file']:
            st.write(st.session_state['uploaded_file'].name)
            st.session_state['df'] =load_data(st.session_state.uploaded_file)
            #st.session_state['df'].iloc[first_idx[0]:, first_idx[1]:].replace(' ', pd.NA, inplace=True)
            #st.session_state['df'].iloc[first_idx[0]:, first_idx[1]:].dropna(inplace = True)
            # 문자열이 포함된 위치 찾기
            string_values = find_string_values(st.session_state['df'], first_idx)
            # 문자열이 포함된 값을 NA로 대체
            replace_string_with_na(st.session_state['df'], string_values)
            # 사용 예시
            st.session_state['df'], last_valid_index = slice_until_first_non_nan_row(st.session_state['df'])
            st.write(string_values)
            st.session_state['mid_ID_idx'] = get_mid_ID_idx(st.session_state['df'], first_idx)
            st.session_state['df'].iloc[first_idx[0]:, first_idx[1]:] = st.session_state['df'].iloc[first_idx[0]:, first_idx[1]:].apply(pd.to_numeric, errors='coerce')
            if subplus_edit:
                st.session_state['df']=st.session_state['df'].iloc[:-1]

    if 'df' in st.session_state:
        uploaded_matrix_X = get_submatrix_withlabel(st.session_state['df'], first_idx[0], first_idx[1], st.session_state['mid_ID_idx'][0], st.session_state['mid_ID_idx'][1], first_idx, numberoflabel=number_of_label)
        uploaded_matrix_R = get_submatrix_withlabel(st.session_state['df'], st.session_state['mid_ID_idx'][0], first_idx[1], st.session_state['df'].shape[0]-1, st.session_state['mid_ID_idx'][1], first_idx, numberoflabel=number_of_label)
        uploaded_matrix_C = get_submatrix_withlabel(st.session_state['df'], first_idx[0], st.session_state['mid_ID_idx'][1], st.session_state['mid_ID_idx'][0], st.session_state['df'].shape[1]-1, first_idx, numberoflabel=number_of_label)
        with st.sidebar.expander("최초 업로드 원본 파일"):
            donwload_data(st.session_state['df'], 'uploaded_df')
            donwload_data(uploaded_matrix_X, 'uploaded_matrix_X')
            donwload_data(uploaded_matrix_R, 'uploaded_matrix_R')
            donwload_data(uploaded_matrix_C, 'uploaded_matrix_C')
        # 원본 부분 header 표시
        st.header('최초 업로드 된 Excel파일 입니다.')
        # 데이터프레임 표시 
        tab1, tab2, tab3, tab4 = st.tabs(['uploaded_df', 'uploaded_matrix_X', 'uploaded_matrix_R', 'uploaded_matrix_C'])
        with tab1:
            st.write(st.session_state['df'])
        with tab2:
            st.write(uploaded_matrix_X)
        with tab3:
            st.write(uploaded_matrix_R)
        with tab4:
            st.write(uploaded_matrix_C)

        if 'df_editing' not in st.session_state:
            st.session_state['df_editing'] = st.session_state['df'].copy()

    if 'data_editing_log' not in st.session_state:
        st.session_state['data_editing_log'] = ''

    if 'df_editing' in st.session_state:
        st.header("DataFrame을 수정합니다.")
        col1, col2, col3 = st.columns(3)
        with col1:
            new_code = st.text_input('새로 삽입할 산업의 code를 입력하세요')
        with col2:
            name = st.text_input('새로 삽입할 산업의 이름을 입력하세요')
        with col3:
            if st.button('Insert'):
                result = insert_row_and_col(st.session_state['df_editing'], first_idx, st.session_state['mid_ID_idx'], new_code, name, number_of_label)
                st.session_state['df_editing'], st.session_state['mid_ID_idx'] = result[0:2]
                st.session_state['data_editing_log'] += (result[2] + '\n\n')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            origin_code = st.text_input('from')
        with col2:
            target_code = st.text_input('to')
        with col3:
            alpha = float(st.text_input('alpha value (0.000 to 1.000)', '0.000'))
        with col4:
            if st.button('Edit Data'):
                result = transfer_to_new_sector(st.session_state['df_editing'], first_idx, origin_code, target_code, alpha)
                st.session_state['df_editing'] = result[0]
                st.session_state['data_editing_log'] += (result[1] + '\n\n')
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button('0인 행(열) 삭제'):
                result = remove_zero_series(st.session_state['df_editing'], first_idx, st.session_state['mid_ID_idx'])
                st.session_state['df_editing'] = result[0]
                st.session_state['data_editing_log'] += (result[1] + '\n\n')
                st.session_state['mid_ID_idx'] = result[2]
        with col2:
             if st.button('-값 절반으로 줄이기'):
                mid_ID_idx_reduced = (st.session_state['mid_ID_idx'][0] - 1, st.session_state['mid_ID_idx'][1] - 1)
                result = reduce_negative_values(st.session_state['df_editing'], first_idx, mid_ID_idx_reduced)
                st.session_state['df_editing'] = result[0]
                st.session_state['data_editing_log'] += (result[1] + '\n\n')
                st.session_state['number_of_divide'] +=1
        with col3:
            if st.button('적용'):
                st.session_state['df_edited'] = st.session_state['df_editing'].copy()
        st.markdown(f"##### - 값 나누는 것: **{st.session_state['number_of_divide']}** 번 적용")
        st.write(st.session_state['df_editing'])
    if 'df_edited' in st.session_state:
        st.header('수정 된 Excel파일 입니다.')
        edited_matrix_X = get_submatrix_withlabel(st.session_state['df_edited'], first_idx[0],first_idx[1], st.session_state['mid_ID_idx'][0], st.session_state['mid_ID_idx'][1], first_idx, numberoflabel = 2)
        edited_matrix_R = get_submatrix_withlabel(st.session_state['df_edited'], st.session_state['mid_ID_idx'][0],first_idx[1], st.session_state['df_edited'].shape[0]-1, st.session_state['mid_ID_idx'][1], first_idx, numberoflabel = 2)
        edited_matrix_C = get_submatrix_withlabel(st.session_state['df_edited'], first_idx[0], st.session_state['mid_ID_idx'][1], st.session_state['mid_ID_idx'][0], st.session_state['df_edited'].shape[1]-1, first_idx, numberoflabel = 2)
        with st.sidebar.expander("수정된 파일"):
            donwload_data(st.session_state['df_edited'], 'edited_df')
            donwload_data(edited_matrix_X, 'edited_matrix_X')
            donwload_data(edited_matrix_R, 'edited_matrix_R')
            donwload_data(edited_matrix_C, 'ueditedmatrix_C')
        # 데이터프레임 표시
        tab1, tab2, tab3, tab4 = st.tabs(['edited_df', 'edited_matrix_X', 'edited_matrix_R', 'edited_matrix_C'])

        with tab1:
            st.write(st.session_state['df_edited'])

        with tab2:
            st.write(edited_matrix_X)

        with tab3:
            st.write(edited_matrix_R)

        with tab4:
            st.write(edited_matrix_C)
        st.header("DataFrame을 임계값을 기준으로 filtering 합니다.")
        st.subheader('threshold에 따른 생존비율 그래프')

    if 'df_edited' in st.session_state:
        st.session_state['df_for_leontief'] = edited_matrix_X.iloc[:-1, :-1].copy()
        st.session_state['df_for_leontief'].index = range(st.session_state['df_for_leontief'].shape[0])
        st.session_state['df_for_leontief'].columns = range(st.session_state['df_for_leontief'].shape[1])
        st.session_state['normalization_denominator'] = st.session_state['df_edited'].iloc[st.session_state['df_edited'].shape[0]-1, first_idx[1]:st.session_state['mid_ID_idx'][1]]
        st.session_state['normalization_denominator'] = pd.to_numeric(st.session_state['normalization_denominator'])
        st.session_state['normalization_denominator_replaced'] = st.session_state['normalization_denominator'].replace(0, np.finfo(float).eps)
        
    if 'df_for_leontief' in st.session_state:
        st.session_state['df_for_leontief_without_label'] = st.session_state['df_for_leontief'].iloc[2:, 2:].copy()
        st.session_state['df_for_leontief_with_label'] = st.session_state['df_for_leontief'].copy()
        tmp = st.session_state['df_for_leontief_without_label'].copy()
        tmp = tmp.apply(pd.to_numeric, errors='coerce')
        tmp = tmp.divide(st.session_state['normalization_denominator_replaced'], axis=1)
        st.session_state['df_for_leontief_with_label'].iloc[2:, 2:] = tmp
        st.session_state['df_normalized_with_label'] = st.session_state['df_for_leontief_with_label'].copy()
        unit_matrix = np.eye(tmp.shape[0])
        subtracted_matrix = unit_matrix - tmp
        leontief = np.linalg.inv(subtracted_matrix.values)
        leontief = pd.DataFrame(leontief)
        # 현재 DataFrame을 가져오기
        current_df = st.session_state['df_for_leontief_with_label']

        # 기존 DataFrame에서 2행과 2열을 제거한 후, 크기를 정의
        existing_rows = current_df.shape[0] - 2  # 기존 DataFrame의 행 수
        existing_cols = current_df.shape[1] - 2  # 기존 DataFrame의 열 수

        # leontief 배열의 크기
        leontief_rows, leontief_cols = leontief.shape

        # 새로운 DataFrame 생성 (NaN으로 초기화)
        new_df = pd.DataFrame(np.nan, index=range(existing_rows + 1), columns=range(existing_cols + 1))

        # leontief 배열이 기존 크기와 일치할 때
        if leontief_rows == existing_rows and leontief_cols == existing_cols:
            # leontief 데이터를 새로운 DataFrame의 적절한 부분에 삽입
            new_df.iloc[:existing_rows, :existing_cols] = leontief  # 기존 데이터 부분에 할당

        # N*N 배열에서 N+1*N+1로 변환
        leontief_with_sums = np.zeros((leontief_rows + 1, leontief_cols + 1))
        leontief_with_sums[:-1, :-1] = leontief  # 기존 leontief 배열을 넣기
        leontief_with_sums[-1, :-1] = leontief.sum(axis=0)  # 마지막 행에 각 열의 합
        leontief_with_sums[:-1, -1] = leontief.sum(axis=1)  # 마지막 열에 각 행의 합

        # 최종적으로 N+1*N+1 배열을 새로운 DataFrame에 업데이트
        # 새로운 크기로 DataFrame을 초기화합니다.
        new_df = pd.DataFrame(leontief_with_sums)
        # 기존 DataFrame의 크기를 늘리기 (NaN으로 초기화)
        current_df = current_df.reindex(index=range(existing_rows + new_df.shape[0]), 
                                        columns=range(existing_cols + new_df.shape[1]))


        # 새로운 DataFrame을 기존 DataFrame의 적절한 위치에 업데이트
        current_df.iloc[2:2 + new_df.shape[0], 2:2 + new_df.shape[1]] = new_df

        # 세션 상태에 업데이트
        st.session_state['df_for_leontief_with_label'] = current_df
        threshold_count(st.session_state['df_for_leontief_with_label'].iloc[2:, 2:])

        st.subheader('Leontief 과정 matrices')
        col1, col2, col3, col4 = st.tabs(['edited', 'normailization denominator', 'normalized', 'leontief inverse'])
        with col1:
            st.write(st.session_state['df_for_leontief'])
        with col2:
            st.write(st.session_state['normalization_denominator'])
        with col3:
            st.write(st.session_state['df_normalized_with_label'])
        with col4:
            st.write(st.session_state['df_for_leontief_with_label'])
            invalid_positions = []
        # 1. 행렬을 순회하며 -0.1 ~ 2 범위를 벗어난 값의 위치를 찾음
        for i in range(leontief.shape[0]):
            for j in range(leontief.shape[1]):
                value = leontief.iloc[i, j]
                if not (-0.1 <= value <= 2):
                    invalid_positions.append((i + 2, j + 2, value))  # 위치 조정 (+2)

        # 2. 대각 원소 중 1 이하인 값의 위치와 값 저장
        diagonal_invalid_positions = []
        for i in range(leontief.shape[0]):
            value = leontief.iloc[i, i]
            if value < 1:
                diagonal_invalid_positions.append((i + 2, i + 2, value))  # 위치 조정 (+2)

        # 결과 출력
        if invalid_positions:
            st.write("조건(-0.1 ~ 2.0)에 맞지 않는 위치와 값:")
            for pos in invalid_positions:
                st.write(f"위치: {pos[:2]}, 값: {pos[2]}")
        else:
            st.write("모든 값이 -0.1 ~ 2 사이의 조건을 만족합니다.")

        # 대각 원소 조건 확인 및 결과 출력
        if diagonal_invalid_positions:
            st.write("대각 원소 중 1 미만인 값이 있습니다:")
            for pos in diagonal_invalid_positions:
                st.write(f"위치: {pos[:2]}, 값: {pos[2]}")
        else:
            st.write("모든 대각 원소가 1보다 큽니다.")
        with st.sidebar.expander('normalized, leontief inverse'):
            donwload_data(st.session_state['df_normalized_with_label'], 'normalized')
            donwload_data(st.session_state['df_for_leontief_with_label'], 'leontief inverse')
        col1, col2= st.columns(2)
        with col1:
            threshold = st.number_input('threshold를 입력하세요', 0.000, 1.000, step=0.001)
        with col2:
            if st.button('Apply threshold'):
                st.session_state.threshold = threshold

    if 'threshold' in st.session_state:
        # binary matrix 생성
        binary_matrix = make_binary_matrix(st.session_state['df_for_leontief_with_label'].iloc[2:, 2:].apply(pd.to_numeric, errors='coerce'), st.session_state.threshold)
        filtered_matrix_X = st.session_state['df_for_leontief'].copy()
        filtered_matrix_X.iloc[2:, 2:] = filtered_matrix_X.iloc[2:, 2:].apply(pd.to_numeric, errors='coerce')*binary_matrix
        filtered_normalized = st.session_state['df_normalized_with_label']
        filtered_normalized.iloc[2:, 2:] = st.session_state['df_normalized_with_label'].iloc[2:, 2:].apply(pd.to_numeric, errors='coerce')*binary_matrix
        filtered_leontief = st.session_state['df_for_leontief_with_label']
        filtered_leontief.iloc[2:, 2:] = st.session_state['df_for_leontief_with_label'].iloc[2:, 2:].apply(pd.to_numeric, errors='coerce')*binary_matrix
        st.subheader('Filtered matrices')
        col1, col2, col3, col4 = st.tabs(['binary_matrix', 'normailization denominator', 'filtered_normalized', 'filtered_leontief'])
        with col1:
            st.write(binary_matrix)
        with col2:
            st.write(filtered_matrix_X)
        with col3:
            st.write(filtered_normalized)
        with col4:
            st.write(filtered_leontief)

        with st.sidebar.expander("filtered file"):
            donwload_data(binary_matrix, 'binary_matrix')
            donwload_data(filtered_matrix_X, 'filtered_matrix_X')
            donwload_data(filtered_normalized, 'filtered_normalized')
            donwload_data(filtered_leontief, 'filtered_leontief')
    st.sidebar.header('수정내역')
    with st.sidebar.expander('수정내역 보기'):
        st.write(st.session_state['data_editing_log'])

if __name__ == "__main__":
    main()
