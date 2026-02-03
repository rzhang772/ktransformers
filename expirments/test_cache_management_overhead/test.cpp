#include <iostream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <random>
#include <cstdint>

// 假设这是 MOE 类中的静态函数，或直接作为自由函数
int* get_new_cache_ids_v1(const uint64_t* cached_expert, const uint64_t* pred_expert,
                          const uint64_t* expert_frequency, int cache_num,
                          int pred_num, int prefetch_num) {
    // 拷贝 cache
    std::vector<uint64_t> new_cache(cached_expert, cached_expert + cache_num);

    // pred 中的非重复元素（且不在 cache 中）
    std::unordered_set<uint64_t> cache_set(new_cache.begin(), new_cache.end());
    std::vector<uint64_t> pred_unique;
    for (int i = 0; i < pred_num; i++) {
        if (cache_set.find(pred_expert[i]) == cache_set.end()) {
            pred_unique.push_back(pred_expert[i]);
        }
    }

    // cache 中的非重复候选（可被替换：即不在 pred 中）
    std::unordered_set<uint64_t> pred_set(pred_expert, pred_expert + pred_num);
    std::vector<std::pair<uint64_t, int>> cache_candidates;
    for (int i = 0; i < cache_num; i++) {
        if (pred_set.find(cached_expert[i]) == pred_set.end()) {
            cache_candidates.push_back({cached_expert[i], (int)expert_frequency[cached_expert[i]]});
        }
    }

    // 按频率升序排序（频率低的优先替换）
    std::sort(cache_candidates.begin(), cache_candidates.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });

    int replace_cnt = std::min(prefetch_num,
                               (int)std::min(pred_unique.size(), cache_candidates.size()));

    for (int i = 0; i < replace_cnt; i++) {
        uint64_t old_id = cache_candidates[i].first;
        uint64_t new_id = pred_unique[i];
        // 找到 old_id 在 cache 中的位置并替换
        for (int j = 0; j < cache_num; j++) {
            if (new_cache[j] == old_id) {
                new_cache[j] = new_id;
                break;
            }
        }
    }

    int* result = new int[cache_num];
    for (int i = 0; i < cache_num; i++) {
        result[i] = (int)new_cache[i];
    }
    return result;
}

// 生成随机数据
void generate_test_data(
    std::vector<uint64_t>& cached_expert,
    std::vector<uint64_t>& pred_expert,
    std::vector<uint64_t>& expert_frequency,
    int cache_num, int pred_num, uint64_t max_expert_id
) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> id_dist(0, max_expert_id - 1);
    std::uniform_int_distribution<uint64_t> freq_dist(1, 10000);

    cached_expert.clear();
    pred_expert.clear();
    expert_frequency.assign(max_expert_id, 0);

    for (int i = 0; i < cache_num; ++i) {
        uint64_t id = id_dist(gen);
        cached_expert.push_back(id);
    }

    for (int i = 0; i < pred_num; ++i) {
        uint64_t id = id_dist(gen);
        pred_expert.push_back(id);
    }

    for (uint64_t i = 0; i < max_expert_id; ++i) {
        expert_frequency[i] = freq_dist(gen);
    }
}

int main() {
    // 测试参数（可调整）
    const int cache_num = 8;      // 缓存大小
    const int pred_num = 8;       // 预测专家数量
    const int prefetch_num = 2;    // 预取数量
    const uint64_t max_expert_id = 256; // 专家 ID 范围 [0, max_expert_id)

    std::vector<uint64_t> cached_expert, pred_expert, expert_frequency;

    generate_test_data(cached_expert, pred_expert, expert_frequency,
                       cache_num, pred_num, max_expert_id);

    // 计时
    auto start = std::chrono::high_resolution_clock::now();

    int* result = get_new_cache_ids_v1(
        cached_expert.data(),
        pred_expert.data(),
        expert_frequency.data(),
        cache_num,
        pred_num,
        prefetch_num
    );

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    std::cout << "Function execution time: " << duration.count() << " ns (" 
              << duration.count() / 1e6 << " ms)\n";

    // 可选：验证结果（例如打印前几个）
    std::cout << "First 10 new cache IDs: ";
    for (int i = 0; i < std::min(10, cache_num); ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << "\n";

    delete[] result;  // 释放内存

    return 0;
}