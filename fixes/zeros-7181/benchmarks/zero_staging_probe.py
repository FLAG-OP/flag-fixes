# Round-6 diagnostic: root-cause the zeros/zero_ complex f_target overhead
# (review 6784122575). Native-vs-registered attribution: the fix's complex
# zero_ path redispatches to x.copy_(0); native ATen complex scalar copy
# itself stages through a full-tensor pageable HtoD (~9-10 GB/s), which the
# timing signature matches. Run WITHOUT flag_gems on PYTHONPATH (native).
import torch, time
torch.manual_seed(0)
def bench(fn, iters=50):
    fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(iters): fn()
    ev1.record(); torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / iters

x = torch.randn(1024, 1024, device="cuda").to(torch.complex64)  # 8MB
xc = x.contiguous()
print("== NATIVE (no flag_gems), c64 2^20 = 8MB ==")
print("native x.copy_(0)          : %.4f ms" % bench(lambda: xc.copy_(0)))
print("native x.zero_()           : %.4f ms" % bench(lambda: xc.zero_()))
c0 = torch.complex(torch.tensor(0.0), torch.tensor(0.0))
print("native copy_(cpu-0dim c64) : %.4f ms" % bench(lambda: xc.copy_(c0)))
print("native copy_(device-0dim)  : %.4f ms" % bench(lambda: xc.copy_(c0.cuda())))
z8 = torch.zeros(8, 8, dtype=torch.complex64, device="cuda")
print("native torch.zeros((32,32), c64) creation: %.4f ms" % bench(lambda: torch.zeros((1024,1024), dtype=torch.complex64, device="cuda")))
