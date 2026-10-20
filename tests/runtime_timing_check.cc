#include "runtime_csv.h"
#include <cassert>
#include <thread>
#include <numeric>
#include <iostream>

using namespace XPLSLAM;
int main(int argc, char **argv) {
    assert(argc == 2);
    RuntimeFrame off;
    { RuntimeCapture capture(nullptr); RuntimePhaseScope phase(RuntimePhase::Pose); }
    assert(runtime_frame == nullptr);
    RuntimeFrame frame;
    RuntimeCapture capture(&frame);
    auto begin = RuntimeFrame::Clock::now();
    capture.Begin(begin);
    {
        RuntimePhaseScope parent(RuntimePhase::BaseFeatures);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        { RuntimePhaseScope child(RuntimePhase::Matching);
          std::this_thread::sleep_for(std::chrono::milliseconds(2)); }
        std::thread other([] { assert(runtime_frame == nullptr); RuntimePhaseScope phase(RuntimePhase::Pose); });
        other.join();
        { RuntimeReasonScope reason(RuntimeLineReason::Relocalization);
          RuntimeReasonScope nested(RuntimeLineReason::WeakPoints);
          assert(runtime_frame->reason == RuntimeLineReason::Relocalization); }
        assert(runtime_frame->reason == RuntimeLineReason::Unspecified);
    }
    auto end=RuntimeFrame::Clock::now(); capture.End(end);
    const double total=std::chrono::duration<double,std::milli>(end-begin).count();
    assert(std::abs(std::accumulate(frame.phase_ms.begin(),frame.phase_ms.end(),0.0)-total)<1e-9);
    assert(frame.phase_ms[1]==0 && frame.phase_ms[3]==0 && frame.phase_ms[4]==0);
    assert(frame.phase_ms[0]>0 && frame.phase_ms[2]>0 && frame.phase_ms[5]>=0);
    assert(frame.calls.empty());
    std::filesystem::create_directories(argv[1]);
    RuntimeCsv csv(argv[1]);
    csv.Append(0,0,frame,total,0,0);
    csv.Finish(1,1,0,total/1000,0,0);
    std::cout << "PASS: nested exclusive accounting, unused zero, disabled capture, thread isolation, reason priority, CSV closure\n";
}
