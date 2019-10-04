import sys
from time import time
import math
import numpy as np
from sklearn.cluster import KMeans

np.random.seed(6)

if len(sys.argv) != 4:
    print("Usage: ./bin/spark-submit Mansi_Ganatra_bfr.py <input_file_path> <n_cluster> <output_file_path>")
    exit(-1)
else:
    input_file_path = sys.argv[1]
    n_clusters = int(sys.argv[2])
    output_file_path = sys.argv[3]


# def update_statistics_for_DS(init_data, original_clusters_indices):
#     global ds_set
#     global ds_summarized_dictionary
#
#     init_data = np.hstack([init_data, np.empty(shape=(init_data.shape[0], 1))])
#
#     for label, cluster in original_clusters_indices.items():
#         N = len(cluster)
#         SUM = init_data[cluster].sum(axis=0)
#         SUMSQ = np.sum(np.square(init_data[cluster].astype(np.float64)), axis =0)
#
#         existing_stats = ds_summarized_dictionary[label]
#
#         updated_N = existing_stats[0] + N
#         updated_SUM = existing_stats[1] + SUM
#         updated_SUMSQ = existing_stats[2] + SUMSQ
#
#         ds_summarized_dictionary.update({label:(updated_N, updated_SUM, updated_SUMSQ)})
#         init_data[cluster,-1] = label
#         ds_set = np.vstack([ds_set, init_data[cluster]])
#
#     return

def generate_statistics_for_initial_DS(init_data, original_clusters_indices):
    global ds_set
    global ds_summarized_dictionary

    # init_data = np.hstack([init_data, np.empty(shape=(init_data.shape[0], 1))])

    # print("Generate stats:")
    # s1= time()

    for label, cluster in original_clusters_indices.items():
        N = len(cluster)
        SUM = np.delete(init_data[cluster], [0,1], axis=1).sum(axis=0)
        SUMSQ = np.sum(np.square(np.delete(init_data[cluster], [0,1], axis=1).astype(np.float64)), axis=0)

        centroid = SUM/N
        variance = (SUMSQ/N)-((SUM/N)**2)
        std_dev = (variance**0.5)

        ds_summarized_dictionary.update({label:(N, SUM, SUMSQ, centroid, variance, std_dev)})
        init_data[cluster,1] = label
        # ds_set = np.vstack([ds_set, init_data[cluster]])
        ds_set += init_data[cluster].tolist()

    # print("Time: ", time()-s1)

    return

def initialize_data(init_data, n_clusters):

    # s1=time()
    n_initial_clusters = 10 * n_clusters
    # seed = 5
    # init_data_copy = init_data.copy()
    current_clusters_indices = {}

    X= init_data[:,2:]
    t1=time()
    kmeans = KMeans(n_clusters=n_initial_clusters).fit(X)
    current_clusters_indices = {label: np.where(kmeans.labels_ == label)[0] for label in range(kmeans.n_clusters)}

    # print("Running first kmeans took: ", time()-t1)
    global rs_set
    # removed_cluster_labels = []
    for label, cluster in current_clusters_indices.items():
        if len(cluster) <= 10:
            # element = init_data[current_clusters_indices[label]]
            # element[1] =-1
            # rs_set.append(element)
            rs_set = np.vstack([rs_set, init_data[current_clusters_indices[label]]])
            rs_set[:,1] = -1
            init_data = np.delete(init_data, current_clusters_indices[label], axis=0)

    new_X= init_data[:,2:]
    # t2=time()
    original_kmeans = KMeans(n_clusters=n_clusters).fit(new_X)
    original_clusters_indices = {label: np.where(original_kmeans.labels_ == label)[0] for label in range(original_kmeans.n_clusters)}
    # print("Running second kmeans took: ", time()-t2)
    # print("Initialized: ", time()-s1)

    generate_statistics_for_initial_DS(init_data, original_clusters_indices)

    return

def generate_CS():

    global rs_set
    global cs_set
    global n_clusters
    global cs_summarized_dictionary

    # seed = 5

    # s1=time()
    if len(rs_set) <=10:
        return

    large_k = 10*n_clusters

    if len(rs_set) <= large_k:
        large_k =int(0.8*len(rs_set))

    X = rs_set[:,2:]
    cs_kmeans = KMeans(n_clusters=large_k).fit(X)
    cs_current_clusters_indices = {n_clusters+1+label: np.where(cs_kmeans.labels_ == label)[0]
                                   for label in range(cs_kmeans.n_clusters)}

    # n_columns = rs_set.shape[1]
    # new_rs = np.array([], dtype=np.float64).reshape(0, n_columns)

    removed_clusters = []
    for label, cluster in cs_current_clusters_indices.items():
        # if len(cluster) == 1:
        #     new_rs = np.vstack([new_rs, rs_set[cs_current_clusters_indices[label]]])
        if len(cluster) != 1:
            # cs_set = np.vstack([cs_set, rs_set[cs_current_clusters_indices[label]]])

            N = len(cluster)
            SUM = np.delete(rs_set[cluster], [0, 1], axis=1).sum(axis=0)
            SUMSQ = np.sum(np.square(np.delete(rs_set[cluster], [0, 1], axis=1).astype(np.float64)), axis=0)

            centroid = SUM / N
            variance = (SUMSQ / N) - ((SUM / N) ** 2)
            std_dev = (variance**0.5)

            cs_summarized_dictionary.update({label: (N, SUM, SUMSQ, centroid, variance, std_dev)})
            rs_set[cluster, 1] = label
            # cs_set = np.vstack([cs_set, rs_set[cluster]])
            cs_set += rs_set[cluster].tolist()

            removed_clusters.append(cluster)

    if len(removed_clusters) > 0:
        rs_set = np.delete(rs_set, np.concatenate(removed_clusters).ravel(), axis=0)

    # rs_set = new_rs
    # print("Initial CS: ", time()-s1)
    return

def calculate_md(point, centroid, std_dev):
    # std_dev = (variance ** 0.5)
    # s1=time()
    md = np.sqrt(np.sum(np.square((point-centroid)/std_dev)))
    # print("Calculating MD: ", time()-s1)
    return md


def process_row(rows):

    global ds_summarized_dictionary
    global threshold
    global cs_summarized_dictionary
    global rs_set
    global cs_set
    global ds_set

    # unassigned_set = np.array([], dtype=np.float64).reshape(0, n_columns)

    unassigned_set = []
    # s1= time()
    for row in rows:
        min_md = math.inf
        min_label = ""

        # print("Point Processing start: ")
        # p_start = time()
        point = row[2:]
        # s11 = time()
        for label, summary in ds_summarized_dictionary.items():
            current_centroid = summary[3]
            current_stddev = summary[5]
            current_md = calculate_md(point, current_centroid, current_stddev)

            if current_md < min_md:
                min_md = current_md
                min_label = label
        # print("Calculating min md: ", time()-s11)

        if min_md < threshold:
            # s12 = time()
            row[1] = min_label

            min_summary = ds_summarized_dictionary[min_label]
            updated_n = min_summary[0] + 1
            updated_sum = min_summary[1] + point
            updated_sumsq = min_summary[2] + np.square(point.astype(np.float64))
            updated_centroid = updated_sum / updated_n
            updated_variance = (updated_sumsq / updated_n) - ((updated_sum / updated_n) ** 2)
            updated_stddev = (updated_variance ** 0.5)
            ds_summarized_dictionary.update({min_label: (updated_n, updated_sum, updated_sumsq, updated_centroid,
                                                         updated_variance, updated_stddev)})
            # ds_set = np.vstack([ds_set, row])
            ds_set.append(row)
            # print("updation: ", time()-s12)
        else:
            unassigned_set.append(row)
    # print("First foor loop: ", time()-s1)
    # s2=time()
    for row in unassigned_set:
        # print("DS check took: ", time()-p_start)
        cs_min_md = math.inf
        cs_min_label = ""
        for cs_label, cs_summary in cs_summarized_dictionary.items():
            cs_current_centroid = cs_summary[3]
            cs_current_stddev = cs_summary[5]
            cs_current_md = calculate_md(point, cs_current_centroid, cs_current_stddev)

            if cs_current_md < cs_min_md:
                cs_min_md = cs_current_md
                cs_min_label = cs_label

        if cs_min_md < threshold:
            row[1] = cs_min_label

            cs_min_summary = cs_summarized_dictionary[cs_min_label]
            cs_updated_n = cs_min_summary[0] + 1
            cs_updated_sum = cs_min_summary[1] + point
            cs_updated_sumsq = cs_min_summary[2] + np.square(point.astype(np.float64))
            cs_updated_centroid = cs_updated_sum / cs_updated_n
            cs_updated_variance = (cs_updated_sumsq / cs_updated_n) - ((cs_updated_sum / cs_updated_n) ** 2)
            cs_updated_stddev = (cs_updated_variance ** 0.5)
            cs_summarized_dictionary.update(
                {min_label: (cs_updated_n, cs_updated_sum, cs_updated_sumsq, cs_updated_centroid, cs_updated_variance, cs_updated_stddev)})
            # cs_set = np.vstack([cs_set, row])
            cs_set.append(row)

        else:
            # print("CS check took: ", time()-p_start)
            row[1] = -1
            rs_set = np.vstack([rs_set, row])
    # print("Second for loop: ", time()-s2)
    # print("Time taken for point: ", time() - p_start)
    return

def merge_new_clusters_to_CS(new_cs_set, new_cs_summarized_dictionary):
    global rs_set
    global cs_set
    global cs_summarized_dictionary

    cs_set_copy = np.array(cs_set)
    new_cs_set_copy = np.array(new_cs_set)

    # print("Merging new clusters to existing CS: ")

    # merge_start = time()

    for new_cs_label, new_cs_summary in new_cs_summarized_dictionary.items():
        min_md = math.inf
        min_label = ""
        new_cs_centroid = new_cs_summary[3]
        # new_cs_variance = new_cs_summary[4]

        for cs_label, cs_summary in cs_summarized_dictionary.items():
            cs_centroid = cs_summary[3]
            cs_stddev = cs_summary[5]
            current_md = calculate_md(new_cs_centroid, cs_centroid, cs_stddev)
            if current_md < min_md:
                min_md = current_md
                min_label = cs_label

        if min_md < threshold:
            min_summary = cs_summarized_dictionary[min_label]
            updated_n = min_summary[0] + new_cs_summary[0]
            updated_sum = min_summary[1] + new_cs_summary[1]
            updated_sumsq = min_summary[2] + new_cs_summary[2]
            updated_centroid = updated_sum/updated_n
            updated_variance = (updated_sumsq/updated_n) - ((updated_sum/updated_n)**2)
            updated_stddev = (updated_variance ** 0.5)

            # add_to_cs = np.array([], dtype=np.float64).reshape(0, n_columns)
            add_to_cs = new_cs_set_copy[new_cs_set_copy[:,1] == new_cs_label]
            add_to_cs[:,1] = min_label
            cs_set_copy = np.vstack([cs_set_copy, add_to_cs])
            new_cs_set_copy = new_cs_set_copy[new_cs_set_copy[:, 1] != new_cs_label]

            cs_summarized_dictionary.update({min_label:(updated_n, updated_sum, updated_sumsq, updated_centroid,
                                                        updated_variance, updated_stddev)})

        else:
            cs_set_copy = np.vstack([cs_set_copy, new_cs_set_copy[new_cs_set_copy[:,1] == new_cs_label]])
            cs_summarized_dictionary.update({new_cs_label:new_cs_summary})
            new_cs_set_copy = new_cs_set_copy[new_cs_set_copy[:, 1] != new_cs_label]

    for refined_cs_label, refined_cs_summary in cs_summarized_dictionary.items():
        min_md = math.inf
        min_label = ""
        refined_cs_centroid = refined_cs_summary[3]
        # new_cs_variance = new_cs_summary[4]

        for cs_label, cs_summary in cs_summarized_dictionary.items():
            if refined_cs_label != cs_label:
                cs_centroid = cs_summary[3]
                cs_stddev = cs_summary[5]
                current_md = calculate_md(refined_cs_centroid, cs_centroid, cs_stddev)
                if current_md < min_md:
                    min_md = current_md
                    min_label = cs_label

        if min_md < threshold:
            min_summary = cs_summarized_dictionary[min_label]
            updated_n = min_summary[0] + refined_cs_summary[0]
            updated_sum = min_summary[1] + refined_cs_summary[1]
            updated_sumsq = min_summary[2] + refined_cs_summary[2]
            updated_centroid = updated_sum / updated_n
            updated_variance = (updated_sumsq / updated_n) - ((updated_sum / updated_n) ** 2)
            updated_stddev = (updated_variance ** 0.5)

            # add_to_cs = np.array([], dtype=np.float64).reshape(0, n_columns)
            add_to_cs = cs_set_copy[cs_set_copy[:, 1] == refined_cs_label]
            # add_to_cs[:, 1] = min_label
            # cs_set_copy = np.vstack([cs_set_copy, add_to_cs])
            # new_cs_set_copy = new_cs_set_copy[new_cs_set_copy[:, 1] != refined_cs_label]

            # cs_set_copy[:,0][cs_set_copy[:,0] == refined_cs_label] = min_label
            cs_set_copy[np.where(cs_set_copy[:,0] == refined_cs_label)] = min_label

            cs_summarized_dictionary.update({min_label: (updated_n, updated_sum, updated_sumsq, updated_centroid,
                                                         updated_variance, updated_stddev)})

    cs_set = cs_set_copy.tolist()

    # merge_end = time()
    # print("Duration to merge: ", merge_end-merge_start)
    return


def run_bfr(next_chunk):
    global rs_set
    # seed = 5
    global n_columns
    global cs_summarized_dictionary

    # print("Processing Chunk:")

    # chunk_start = time()


    process_row(next_chunk)

    # chunk_end = time()
    # print("Duration to process chunk: ", chunk_end-chunk_start)

    if len(rs_set) <=10:
        return

    large_k = 10*n_clusters

    if len(rs_set) <= large_k:
        large_k =int(0.8*len(rs_set))

    X = rs_set[:,2:]
    cs_kmeans = KMeans(n_clusters=large_k).fit(X)
    range_start = max(cs_summarized_dictionary.keys())

    cs_current_clusters_indices = {range_start+1+label: np.where(cs_kmeans.labels_ == label)[0]
                                   for label in range(cs_kmeans.n_clusters)}

    # n_columns = rs_set.shape[1]
    # new_rs = np.array([], dtype=np.float64).reshape(0, n_columns)
    # print("Creating new CS clusters: ")
    new_cs_summarized_dictionary ={}
    # new_cs_set = np.array([], dtype=np.float64).reshape(0, n_columns)

    new_cs_set =[]

    removed_clusters = []
    for label, cluster in cs_current_clusters_indices.items():
        # if len(cluster) == 1:
        #     new_rs = np.vstack([new_rs, rs_set[cs_current_clusters_indices[label]]])
        if len(cluster) != 1:
            # cs_set = np.vstack([cs_set, rs_set[cs_current_clusters_indices[label]]])

            N = len(cluster)
            SUM = np.delete(rs_set[cluster], [0, 1], axis=1).sum(axis=0)
            SUMSQ = np.sum(np.square(np.delete(rs_set[cluster], [0, 1], axis=1).astype(np.float64)), axis=0)

            centroid = SUM / N
            variance = (SUMSQ / N) - ((SUM / N) ** 2)
            stddev = (variance ** 0.5)

            new_cs_summarized_dictionary.update({label: (N, SUM, SUMSQ, centroid, variance, stddev)})
            rs_set[cluster, 1] = label
            # new_cs_set = np.vstack([new_cs_set, rs_set[cluster]])
            new_cs_set += rs_set[cluster].tolist()

            removed_clusters.append(cluster)

    if len(removed_clusters) > 0:
        rs_set = np.delete(rs_set, np.concatenate(removed_clusters).ravel(), axis=0)
    # print("Newly created CS clusters: ")
    # print(new_cs_set)
    merge_new_clusters_to_CS(new_cs_set, new_cs_summarized_dictionary)

    return

def merge_CS_to_DS():
    global cs_summarized_dictionary
    global ds_summarized_dictionary
    global ds_set
    global cs_set

    cs_set_copy = np.array(cs_set)
    ds_set_copy =  np.array(ds_set)

    # print("Merging all cs to ds for last round:")
    # m_start = time()
    removed_labels =[]
    for cs_label, cs_summary in cs_summarized_dictionary.items():
        min_md = math.inf
        min_label = ""

        cs_centroid = cs_summary[3]
        # new_cs_variance = new_cs_summary[4]

        for ds_label, ds_summary in ds_summarized_dictionary.items():
            ds_centroid = ds_summary[3]
            ds_stddev = ds_summary[5]
            current_md = calculate_md(cs_centroid, ds_centroid, ds_stddev)
            if current_md < min_md:
                min_md = current_md
                min_label = ds_label

        min_summary = ds_summarized_dictionary[min_label]
        updated_n = min_summary[0] + cs_summary[0]
        updated_sum = min_summary[1] + cs_summary[1]
        updated_sumsq = min_summary[2] + cs_summary[2]
        updated_centroid = updated_sum / updated_n
        updated_variance = (updated_sumsq / updated_n) - ((updated_sum / updated_n) ** 2)
        updated_stddev = (updated_variance ** 0.5)

        add_to_ds = cs_set_copy[cs_set_copy[:, 1] == cs_label]
        add_to_ds[:, 1] = min_label
        ds_set_copy = np.vstack([ds_set_copy, add_to_ds])
        cs_set_copy = cs_set_copy[cs_set_copy[:, 1] != cs_label]
        ds_summarized_dictionary.update({min_label: (updated_n, updated_sum, updated_sumsq, updated_centroid,
                                                     updated_variance, updated_stddev)})

        removed_labels.append(cs_label)

    for label in removed_labels:
        del cs_summarized_dictionary[label]

    cs_set = cs_set_copy.tolist()
    ds_set = ds_set_copy.tolist()

    # m_end = time()
    # print("Duration to merge cs to ds: ", m_end-m_start)
    return


intermediate_result = {}

start_time = time()

# ip= "C:/Users/mansi/PycharmProjects/Mansi_Ganatra_HW5/hw5_clustering.txt"
data = np.loadtxt(input_file_path, delimiter=",")
data_copy = data.copy()
n_sample = len(data)
# print(n_sample)

# cs_set = set()
# rs_set = set()
# ds_set = set()

# global ds_summarized_dictionary
# global cs_set
# global rs_set
# global ds_set

percentage = 0.2
sample_size = int(n_sample*percentage)
# drop the index column
# data = np.delete(data, 0, 1)

init_data = data[:sample_size]
# print(len(init_data))

n_columns = data.shape[1]
# cs_set = np.array([], dtype=np.float64).reshape(0, n_columns)
# rs_set = np.array([], dtype=np.float64).reshape(0, n_columns)
# ds_set = np.array([], dtype=np.float64).reshape(0, n_columns)

# cs_set = np.array([], dtype=np.float64).reshape(0, n_columns)
rs_set = np.array([], dtype=np.float64).reshape(0, n_columns)
# ds_set = np.array([], dtype=np.float64).reshape(0, n_columns)

cs_set = []
ds_set = []

ds_summarized_dictionary = {}
cs_summarized_dictionary = {}

# excluding index, cluster column from data rest are features
dimension = n_columns-2
threshold = 2*(math.sqrt(dimension))

# print("Initialization step: ")
initialize_data(init_data, n_clusters)

generate_CS()

# print("The intermediate results: ")
# print("Round 1: ", len(ds_set), len(cs_summarized_dictionary.keys()), len(cs_set), rs_set.shape[0])
intermediate_result.update({1: (len(ds_set), len(cs_summarized_dictionary.keys()), len(cs_set), rs_set.shape[0])})

start = sample_size
end = start + sample_size

round = 2
while start < n_sample:
    run_bfr(data[start:end])
    start = end
    end = start + sample_size
    # print("The intermediate results: ")
    # print("Round " + str(round) + ": ", len(ds_set), len(cs_summarized_dictionary.keys()), len(cs_set), rs_set.shape[0])
    intermediate_result.update({round: (len(ds_set), len(cs_summarized_dictionary.keys()), len(cs_set), rs_set.shape[0])})
    round += 1

    if start >= n_sample:
        merge_CS_to_DS()

        clustered_data = np.vstack([ds_set, rs_set])
        clustered_data = clustered_data[clustered_data[:,0].argsort()]

        # print("Covered all points?: ", len(clustered_data)==len(data))

        # print("Round " + str(round) + ": ", len(ds_set), len(cs_summarized_dictionary.keys()), len(cs_set),
        #       rs_set.shape[0])
        intermediate_result.update(
            {round: (len(ds_set), len(cs_summarized_dictionary.keys()), len(cs_set), rs_set.shape[0])})

# print("The clustering results:")
# print(clustered_data[:,[0,1]])

with open(output_file_path, "w+", encoding="utf-8") as fp:
    fp.write("The intermediate results:")
    fp.write('\n')
    fp.write('\n'.join('Round {}: {},{},{},{}'.format(l, x[0], x[1], x[2], x[3]) for l,x in intermediate_result.items()))
    fp.write('\n\n')
    fp.write("The clustering results:")
    fp.write('\n')
    fp.write('\n'.join('{},{}'.format(int(x[0]), int(x[1])) for x in clustered_data[:,[0,1]].tolist()))

# print("**************************** DS ***************************")
# print(ds_summarized_dictionary)
#
# print("****************************** CS *************************")
# print(cs_summarized_dictionary)
#
# print("***************************** RS ***************************")
# print(rs_set)
end_time = time()
print("Duration: ", end_time-start_time)

from sklearn.metrics import normalized_mutual_info_score

score = normalized_mutual_info_score(data[:, 1], clustered_data[:,1])

print("Normalized Score: ", score)