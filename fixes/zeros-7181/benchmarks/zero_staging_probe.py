# Round-6/7 diagnostic: root-cause the zeros/zero_ complex f_target overhead
# (reviews 6784122575 + 6785197539). Native-vs-registered attribution: the
# fix's complex zero_ path redispatches to x.copy_(0); native ATen complex
# scalar copy itself stages through a full-tensor pageable HtoD (~9-10 GB/s),
# which the timing signature matches. Run WITHOUT flag_gems on PYTHONPATH
# (native).
#
# Round-7 correction (review 6785197539 P2): the device-scalar control now
# PRE-BUILDS c0_device outside the timed lambda, and reports separately
# (a) device-scalar creation cost, (b) copy-only with the pre-built device
# scalar, (c) native zero_() memset, (d) copy_(0) python scalar (the fix's
# inherited path), (e) copy_ with a CPU 0-dim tensor — so the host-staging
# attribution is not polluted by per-iteration preparation.
import torch

torch.manual_seed(0)


def bench(fn, iters=50):
    fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(iters):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / iters


x = torch.randn(1024, 1024, device="cuda").to(torch.complex64)  # 8MB
xc = x.contiguous()
c0 = torch.complex(torch.tensor(0.0), torch.tensor(0.0))

print("== NATIVE (no flag_gems), c64 2^20 = 8MB ==")
print("[fix's inherited path]")
print("native x.copy_(0) (python scalar)     : %.4f ms"
      % bench(lambda: xc.copy_(0)))
print("native copy_(cpu-0dim c64 tensor)     : %.4f ms"
      % bench(lambda: xc.copy_(c0)))
print("[device-side alternatives]")
# creation cost of the device scalar: single-shot CUDA-event measurement
# (round-8 review: bench() averages warmup+50 iterations, which is NOT a
# one-off cost; both numbers are reported and clearly labelled)
e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
torch.cuda.synchronize()
e0.record()
c0_once = c0.cuda()
e1.record()
torch.cuda.synchronize()
print("device-scalar creation (c0.cuda()), SINGLE-SHOT: %.4f ms"
      % e0.elapsed_time(e1))
c0_device = c0.cuda()
torch.cuda.synchronize()
print("device-scalar creation (c0.cuda()), 50-iter average: %.4f ms"
      % bench(lambda: c0.cuda()))
print("native copy_(PRE-BUILT device-0dim), copy-only: %.4f ms"
      % bench(lambda: xc.copy_(c0_device)))
print("native x.zero_() (device memset)      : %.4f ms"
      % bench(lambda: xc.zero_()))
print("[creation reference]")
print("native torch.zeros(c64) creation      : %.4f ms"
      % bench(lambda: torch.zeros((1024, 1024), dtype=torch.complex64,
                                  device="cuda")))

# c128 confirmation
x128 = torch.randn(1024, 1024, device="cuda").to(torch.complex128)
print("== c128 2^20 = 16MB ==")
print("native x128.copy_(0)                  : %.4f ms"
      % bench(lambda: x128.copy_(0)))
c0d128 = c0.to(torch.complex128).cuda()
torch.cuda.synchronize()
print("native x128.copy_(PRE-BUILT device)   : %.4f ms"
      % bench(lambda: x128.copy_(c0d128)))
print("native x128.zero_()                   : %.4f ms"
      % bench(lambda: x128.zero_()))
