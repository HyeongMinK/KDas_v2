import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import io
import zipfile
import unicodedata

### 자동화 관련 함수 선언
def _nfc(s: str) -> str:
    return unicodedata.normalize('NFC', s)

def _fix_zip_name(name: str) -> str:
    """
    zipfile이 cp437로 잘못 디코딩한 파일명을 복구 시도.
    1) cp437 bytes로 되돌린 뒤
    2) utf-8 / cp949 순으로 decode 시도
    """
    try:
        raw = name.encode("cp437")
    except Exception:
        return name

    for enc in ("utf-8", "cp949"):
        try:
            return raw.decode(enc)
        except Exception:
            pass

    # 최후: cp949로 깨지더라도 replace
    return raw.decode("cp949", errors="replace")


def _pick_excel_from_zip(z: zipfile.ZipFile, original_filename_no_ext: str):
    """ZIP 내부에서 원본 파일명 기반 매칭 -> 실패 시 첫 번째 엑셀 fallback"""

    infos = []
    for info in z.infolist():
        raw = info.filename
        fixed = _nfc(_fix_zip_name(raw)).replace("\\", "/")

        # __MACOSX 제거 + 엑셀만
        if fixed.startswith("__MACOSX") or "/__MACOSX/" in fixed:
            continue
        if not fixed.endswith((".xlsx", ".xls")):
            continue

        infos.append((info, fixed))

    # (표시용) clean name
    clean_names = []
    info_by_clean = {}
    for info, fixed in infos:
        base = fixed.split("/")[-1]
        clean_no_ext = base.rsplit(".", 1)[0]
        clean_names.append(clean_no_ext)
        info_by_clean[clean_no_ext] = info   # ✅ ZipInfo 저장

    # 1) 자동 매칭
    norm_orig = _nfc(original_filename_no_ext)
    for clean in clean_names:
        parts = [x for x in clean.split("_") if x]
        parts = [_nfc(x) for x in parts]
        if parts and all(part in norm_orig for part in parts):
            return clean, info_by_clean[clean], "matched"

    # 2) fallback: 첫 번째 엑셀
    if clean_names:
        clean = clean_names[0]
        return clean, info_by_clean[clean], "fallback_first"

    return None, None, "no_excel"



def prepare_batch_preview(alpha_file, original_filename_no_ext: str):
    """
    1) ZIP이면 매칭 후 batch_df 로드 / 엑셀이면 바로 로드
    2) 텍스트 미리보기 라인 생성
    return: (batch_df, meta, preview_lines, summary_lines)
    """
    meta = {
        "uploaded": alpha_file.name,
        "kind": "zip" if alpha_file.name.endswith(".zip") else "excel",
        "matched_file": None,
        "match_mode": None
    }

    # --- 1단계: 파일 확보 (업로드 즉시 실행) ---
    if alpha_file.name.endswith(".zip"):
        zip_bytes = io.BytesIO(alpha_file.getvalue())
        with zipfile.ZipFile(zip_bytes, 'r') as z:
            matched_clean, matched_info, mode = _pick_excel_from_zip(z, original_filename_no_ext)
            if mode == "no_excel":
                raise ValueError("ZIP 내부에 엑셀 파일이 없습니다.")

            meta["matched_file"] = matched_clean
            meta["match_mode"] = mode

            # ✅ 문자열 경로가 아니라 ZipInfo로 open
            with z.open(matched_info) as f:
                batch_df = pd.read_excel(
                f,
                dtype=str  # <─ 전체를 문자열로 받음 (숫자로 오인 금지)
            )

    else:
        meta["matched_file"] = alpha_file.name
        meta["match_mode"] = "no_match_needed"
        batch_df = pd.read_excel(
            alpha_file,
            dtype=str  # <─ 여기서도 동일
        )

    # --- 검증/정리 ---
    needed_cols = {"from", "to", "to_name", "alpha"}
    if not needed_cols.issubset(batch_df.columns):
        raise ValueError(f"엑셀 파일에 다음 컬럼이 포함되어야 합니다: {needed_cols}")


    df = batch_df.copy()
    df["from"] = df["from"].astype(str)
    df["to"] = df["to"].astype(str)
    df["to_name"] = df["to_name"].astype(str)
    df["to_name"] = df["to_name"].replace("nan", "").fillna("")
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")

    # alpha가 NaN인 행 제거
    df = df.dropna(subset=["alpha"])

    # --- 2단계: 텍스트 미리보기 생성 ---
    preview_lines = []
    for _, r in df.iterrows():
        nm = r["to_name"] if r["to_name"] else "-"
        preview_lines.append(f"{r['from']} -> {r['to']}({nm}) : {float(r['alpha'])*100:.4f}%")

    # from별 합/잔여
    summary_lines = []
    grouped = df.groupby("from")["alpha"].sum()
    for origin_code, total_alpha in grouped.items():
        remaining = 1.0 - float(total_alpha)
        summary_lines.append(
            f"[from={origin_code}] 이동합={float(total_alpha)*100:.4f}%, 잔여={remaining*100:.4f}%"
        )

    return df, meta, preview_lines, summary_lines

### 사용자 정의 함수 선언
def make_binary_matrix(matrix, threshold):
    # 임계값 이하의 원소들을 0으로 설정
    binary_matrix = matrix.apply(lambda x: np.where(x > threshold, 1, 0))
    return binary_matrix

def filter_matrix(matrix, threshold):
    # 임계값 이하의 원소들을 0으로 설정
    filtered_matrix = matrix.where(matrix > threshold, 0)
    return filtered_matrix

def calculate_network_centralities(G_bn, df_label, use_weight=False):
    weight_arg = 'weight' if use_weight else None

    # Degree
    in_degree_bn = dict(G_bn.in_degree(weight=weight_arg))
    out_degree_bn = dict(G_bn.out_degree(weight=weight_arg))

    df_degree = df_label.iloc[2:, :2].copy()
    df_degree["in_degree"] = pd.Series(in_degree_bn).sort_index().values.reshape(-1, 1)
    df_degree["out_degree"] = pd.Series(out_degree_bn).sort_index().values.reshape(-1, 1)

    gd_in_mean = df_degree["in_degree"].mean()
    gd_in_std = df_degree["in_degree"].std()
    gd_out_mean = df_degree["out_degree"].mean()
    gd_out_std = df_degree["out_degree"].std()

    # Betweenness
    bc_bn = nx.betweenness_centrality(G_bn, normalized=False, endpoints=False, weight=weight_arg)
    num_n = len(G_bn)
    bc_bn = {node: value / (num_n * (num_n - 1)) for node, value in bc_bn.items()}

    df_bc = df_label.iloc[2:, :2].copy()
    df_bc["Betweenness Centrality"] = pd.Series(bc_bn).sort_index().values.reshape(-1, 1)

    bc_mean = df_bc["Betweenness Centrality"].mean()
    bc_std = df_bc["Betweenness Centrality"].std()

    # Closeness
    cci_bn = nx.closeness_centrality(G_bn, distance=weight_arg)
    cco_bn = nx.closeness_centrality(G_bn.reverse(), distance=weight_arg)

    df_cc = df_label.iloc[2:, :2].copy()
    df_cc["Indegree_Closeness Centrality"] = pd.Series(cci_bn).sort_index().values.reshape(-1, 1)
    df_cc["Outdegree_Closeness Centrality"] = pd.Series(cco_bn).sort_index().values.reshape(-1, 1)

    cc_in_mean = df_cc["Indegree_Closeness Centrality"].mean()
    cc_in_std = df_cc["Indegree_Closeness Centrality"].std()
    cc_out_mean = df_cc["Outdegree_Closeness Centrality"].mean()
    cc_out_std = df_cc["Outdegree_Closeness Centrality"].std()

    # Eigenvector
    evi_bn = nx.eigenvector_centrality(G_bn, max_iter=500, tol=1e-06, weight=weight_arg)
    evo_bn = nx.eigenvector_centrality(G_bn.reverse(), max_iter=500, tol=1e-06, weight=weight_arg)

    df_ev = df_label.iloc[2:, :2].copy()
    df_ev["Indegree_Eigenvector Centrality"] = pd.Series(evi_bn).sort_index().values.reshape(-1, 1)
    df_ev["Outdegree_Eigenvector Centrality"] = pd.Series(evo_bn).sort_index().values.reshape(-1, 1)

    ev_in_mean = df_ev["Indegree_Eigenvector Centrality"].mean()
    ev_in_std = df_ev["Indegree_Eigenvector Centrality"].std()
    ev_out_mean = df_ev["Outdegree_Eigenvector Centrality"].mean()
    ev_out_std = df_ev["Outdegree_Eigenvector Centrality"].std()

    # HITS (가중치 지원 안 함 → 그대로 사용)
    hubs, authorities = nx.hits(G_bn, max_iter=1000, tol=1e-08, normalized=True)

    df_hi = df_label.iloc[2:, :2].copy()
    df_hi["HITS Hubs"] = pd.Series(hubs).sort_index().values.reshape(-1, 1)
    df_hi["HITS Authorities"] = pd.Series(authorities).sort_index().values.reshape(-1, 1)

    hi_hub_mean = df_hi["HITS Hubs"].mean()
    hi_hub_std = df_hi["HITS Hubs"].std()
    hi_ah_mean = df_hi["HITS Authorities"].mean()
    hi_ah_std = df_hi["HITS Authorities"].std()

    return (
        df_degree, df_bc, df_cc, df_ev, df_hi,
        gd_in_mean, gd_in_std, gd_out_mean, gd_out_std,
        bc_mean, bc_std,
        cc_in_mean, cc_in_std, cc_out_mean, cc_out_std,
        ev_in_mean, ev_in_std, ev_out_mean, ev_out_std,
        hi_hub_mean, hi_hub_std, hi_ah_mean, hi_ah_std
    )


# 임계 값을 0-1까지로, 25%로 x축을 한정해서 시각화, 최대 변화율 지점의 x축 값 찾기
@st.cache_data
def threshold_count(matrix):
    import numpy as np
    import matplotlib.pyplot as plt
    import streamlit as st

    L = matrix
    element_counts = []
    element_ratios = []

    N = L.shape[0]
    total_elements = N**2 - N  # 대각선 제외한 전체 원소 수

    # 임계값 생성
    threshold_values = np.linspace(0, 1, 1000)[:250]

    for threshold in threshold_values:
        thresholded_matrix = filter_matrix(L, threshold)
        thresholded_matrix = thresholded_matrix.copy().to_numpy()
        np.fill_diagonal(thresholded_matrix, 0)

        count = (thresholded_matrix >= threshold).sum().sum()
        ratio = count / total_elements

        element_counts.append(count)
        element_ratios.append(ratio)

    # 최대 변화율(절대값) 찾기 (ratio 기준)
    max_change = 0
    max_change_index = 0
    for i in range(1, len(element_ratios)):
        change = abs(element_ratios[i - 1] - element_ratios[i])
        if change > max_change:
            max_change = change
            max_change_index = i

    max_change_threshold = threshold_values[max_change_index]

    # 그래프 그리기 (이중 y축)
    fig, ax1 = plt.subplots()

    color1 = 'tab:blue'
    ax1.set_xlabel('Threshold Value')
    ax1.set_ylabel('Count (Number of Elements ≥ Threshold)', color=color1)
    ax1.plot(threshold_values, element_counts, color=color1, label='Count')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()  # 두 번째 y축 생성
    color2 = 'tab:red'
    ax2.set_ylabel('Ratio (Count / (N² - N))', color=color2)
    ax2.plot(threshold_values, element_ratios, color=color2, linestyle='--', label='Ratio')
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.tight_layout()
    st.pyplot(fig)
    st.write(f'생존율 급감 구간의 임계 값 : {max_change_threshold:.4f}')

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

    # first_idx에서 mid_ID_idx까지의 범위 슬라이싱
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

    # 수정된 값을 원본 데이터프레임에 다시 반영 (first_idx에서 mid_ID_idx까지의 부분)
    df_editing.iloc[first_idx[0]:mid_ID_idx[0], first_idx[1]:mid_ID_idx[1]] = df_test

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
    button = st.download_button(label=f"{file_name} 다운로드", data=csv, file_name=file_name+".csv", mime='text/csv')
    return button




@st.cache_data()
def load_data(file):
    st.session_state['df'] = pd.read_excel(file, header=None)
    return st.session_state['df']

@st.cache_data 
def convert_df(df):
    return df.to_csv(header=False, index=False).encode('utf-8-sig')


@st.cache_data
def make_zip_bytes(dfs: dict[str, pd.DataFrame]) -> bytes:
    """
    dfs: dict where keys are desired CSV filenames and values are DataFrames.
    반환값: ZIP 파일의 바이너리
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, df in dfs.items():
            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
            zf.writestr(f"{fname}.csv", csv_bytes)
    return buf.getvalue()

def download_multiple_csvs_as_zip(dfs: dict[str, pd.DataFrame], zip_name: str):
    zip_bytes = make_zip_bytes(dfs)
    return st.download_button(
        label=f"{zip_name} 다운로드",
        data=zip_bytes,
        file_name=f"{zip_name}.zip",
        mime="application/zip",
    )

def compute_leontief_inverse(A, epsilon=0.05, max_iter=100):
    """
    Leontief 역행렬을 무한급수(I + A + A^2 + ...)로 근사 계산하는 함수.
    수렴 조건: 누적합의 상대변화가 epsilon 이하가 될 때까지 반복.
    
    Parameters:
        A (ndarray): 투입계수행렬.
        epsilon (float): 수렴 판정 기준 (예: 0.05 = 5%).
        max_iter (int): 최대 반복 횟수 (무한급수의 수렴이 안 될 경우 대비).
    
    Returns:
        M (ndarray): I + A + A^2 + ... + A^r (r번째 항까지 계산한 근사 Leontief 역행렬).
    """
    n = A.shape[0]
    I = np.eye(n)           # n x n 항등행렬 생성
    M = I.copy()            # 초기 누적합: M(0) = I
    s_prev = np.sum(M)      # 초기 전체합 (s(0))
    k = 1                   # 거듭제곱 차수 초기화

    while k < max_iter:
        # A^k 계산 (행렬의 거듭제곱)
        A_power = np.linalg.matrix_power(A, k)
        
        # 누적합 업데이트: M(k) = M(k-1) + A^k
        M_new = M + A_power
        
        # 새로운 전체합 계산
        s_new = np.sum(M_new)
        
        # 상대 변화량 계산: (s(k) - s(k-1)) / s(k-1)
        ratio_change = (s_new - s_prev) / s_prev if s_prev != 0 else 0
        
        # 중간 결과 출력 (디버그용)
        print(f"Iteration {k}: ratio_change = {ratio_change:.4f}")
        
        # 수렴 판정: 상대 변화가 epsilon 이하이면 종료
        if ratio_change <= epsilon:
            M = M_new
            break
        
        # 업데이트 후 다음 반복 진행
        M = M_new
        s_prev = s_new
        k += 1
    
    return M

def separate_diagonals(N0):
    """
    입력 행렬 N0에서 대각원소와 비대각원소(네트워크 base)를 분리.
    
    Parameters:
        N0 (ndarray): Leontief 역행렬 근사 (I + A + A^2 + ...).
    
    Returns:
        Diagon (ndarray): N0에서 대각원소만 남기고 나머지를 0으로 만든 행렬.
        N (ndarray): N0에서 대각원소를 모두 0으로 만든 네트워크 행렬.
    """
    # np.diag: 대각 성분 추출, np.diagflat: 대각 행렬로 재구성
    Diagon = np.diag(np.diag(N0))
    N = N0 - Diagon
    return Diagon, N

def threshold_network(N, delta):
    """
    네트워크 행렬 N에서 임계치 delta보다 작은 값들을 0으로 대체.
    
    Parameters:
        N (ndarray): 원본 네트워크 행렬.
        delta (float): 임계치 값.
    
    Returns:
        N_thresholded (ndarray): thresholding 적용된 네트워크 행렬.
    """
    N_thresholded = N.copy()
    N_thresholded[N_thresholded < delta] = 0
    return N_thresholded

def create_binary_network(N):
    """
    가중치 네트워크 행렬 N를 이진(0-1) 네트워크로 변환 (양수면 1, 아니면 0).
    
    Parameters:
        N (ndarray): 가중치 네트워크 행렬.
    
    Returns:
        BN (ndarray): 이진화된 네트워크 (방향성 유지).
    """
    BN = (N > 0).astype(int)
    return BN

def create_undirected_network(BN):
    """
    방향성이 있는 이진 네트워크 BN를 무방향 네트워크로 변환.
    두 노드 간 어느 한쪽이라도 연결되어 있으면, 무방향 연결로 처리.
    
    Parameters:
        BN (ndarray): 이진화된 방향성 네트워크.
    
    Returns:
        UN (ndarray): 무방향(대칭) 이진 네트워크.
    """
    UN = ((BN + BN.T) > 0).astype(int)
    return UN