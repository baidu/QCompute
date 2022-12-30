#pragma once

#include "common.hpp"

/*
long long clock2_wall() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    assert(ts.tv_nsec < 1000000000);

    return long long(ts.tv_sec) * 1000000000 + long long(ts.tv_nsec);
}

long long clock2_cpu() {
    timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    assert(ts.tv_nsec < 1000000000);

    return long long(ts.tv_sec) * 1000000000 + long long(ts.tv_nsec);
}
*/

long long clock2_wall() {
    boost::chrono::time_point<boost::chrono::process_real_cpu_clock> t = boost::chrono::process_real_cpu_clock::now();
    return t.time_since_epoch().count();
}

long long clock2_cpu() {
    boost::chrono::time_point<boost::chrono::process_system_cpu_clock> t1 = boost::chrono::process_system_cpu_clock::now();
    boost::chrono::time_point<boost::chrono::process_user_cpu_clock> t2 = boost::chrono::process_user_cpu_clock::now();
    return t1.time_since_epoch().count() + t2.time_since_epoch().count();
}

long long g_ticTime = 0;

void tic() {
    g_ticTime = clock2_wall();
}

void toc() {
    cout << (clock2_wall() - g_ticTime) / 1000000000.0 << "s elapsed." << endl;
}

// item: (count, wall-clock time, cpu time)
map<string, tuple<int, long long, long long>> g_profilerTotalTimeCount;

// item: (wall-clock time, cpu time)
map<string, tuple<long long, long long>> g_profilerBeginTime;

void profilerBegin(string item) {
    assert(get<0>(g_profilerBeginTime[item]) == 0 && "g_profilerBeginTime check failed in profilerBegin. (SL.5.14)");
    assert(get<1>(g_profilerBeginTime[item]) == 0 && "g_profilerBeginTime check failed in profilerBegin. (SL.5.14)");

    get<0>(g_profilerBeginTime[item]) = clock2_wall();
    get<1>(g_profilerBeginTime[item]) = clock2_cpu();
}

void profilerEnd(string item) {
    assert(get<0>(g_profilerBeginTime[item]) > 0 && "g_profilerBeginTime check failed in profilerEnd. (SL.5.15)");
    assert(get<1>(g_profilerBeginTime[item]) > 0 && "g_profilerBeginTime check failed in profilerEnd. (SL.5.15)");

    get<0>(g_profilerTotalTimeCount[item])++;
    get<1>(g_profilerTotalTimeCount[item]) += clock2_wall() - get<0>(g_profilerBeginTime[item]);
    get<2>(g_profilerTotalTimeCount[item]) += clock2_cpu() - get<1>(g_profilerBeginTime[item]);

    get<0>(g_profilerBeginTime[item]) = 0;
    get<1>(g_profilerBeginTime[item]) = 0;
}

string profilerGetSummary() {
    for (auto &itemBeginTime: g_profilerBeginTime) {
        assert(get<0>(itemBeginTime.second) == 0 &&
               "g_profilerBeginTime check failed in profilerGetSummary. (SL.5.16)");
        assert(get<1>(itemBeginTime.second) == 0 &&
               "g_profilerBeginTime check failed in profilerGetSummary. (SL.5.16)");
    }

    stringstream ss;
    ss << setw(24) << left << "item" << right
       << setw(6) << "count"
       << setw(12) << "W-C time"
       << setw(6) << "%"
       << setw(12) << "CPU time"
       << setw(6) << "%"
       << endl;

    double maxWallClockTime = -1.0;
    double maxCpuTime = -1.0;
    for (auto &itemTotalTimeCount: g_profilerTotalTimeCount) {
        maxWallClockTime = max(maxWallClockTime, get<1>(itemTotalTimeCount.second) / 1000000000.0);
        maxCpuTime = max(maxCpuTime, get<2>(itemTotalTimeCount.second) / 1000000000.0);
    }

    for (auto &itemTotalTimeCount: g_profilerTotalTimeCount) {
        ss << setw(24) << left << itemTotalTimeCount.first.c_str() << right
           << setw(6) << get<0>(itemTotalTimeCount.second)
           << setw(12) << get<1>(itemTotalTimeCount.second) / 1000000000.0
           << setw(6) << int(get<1>(itemTotalTimeCount.second) / 1000000000.0 / maxWallClockTime * 100 + 0.5)
           << setw(12) << get<2>(itemTotalTimeCount.second) / 1000000000.0
           << setw(6) << int(get<2>(itemTotalTimeCount.second) / 1000000000.0 / maxCpuTime * 100 + 0.5)
           << endl;
    }

    return ss.str();
}

void profilerReset() {
    for (auto &itemBeginTime: g_profilerBeginTime) {
        assert(get<0>(itemBeginTime.second) == 0 && "g_profilerBeginTime check failed in profilerReset. (SL.5.17)");
        assert(get<1>(itemBeginTime.second) == 0 && "g_profilerBeginTime check failed in profilerReset. (SL.5.17)");
    }

    g_profilerTotalTimeCount.clear();
}
