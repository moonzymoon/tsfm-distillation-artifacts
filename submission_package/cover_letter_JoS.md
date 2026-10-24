# Cover Letter — The Journal of Supercomputing

Dear Editor-in-Chief,

We submit our manuscript "Distilling Frozen Time-Series Foundation
Model Teachers into Lightweight Edge Anomaly Detectors: A Preregistered
Quality–Fidelity–Cost Study" for consideration in The Journal of
Supercomputing.

Time-series foundation models (TSFMs) such as Chronos-Bolt offer strong
zero-shot anomaly detection, but their per-point inference cost (over
100 ms single-threaded in our measurements) is far beyond edge budgets.
The standard remedy—distill the frozen teacher into a lightweight
student and deploy the student alone—silently assumes the teacher's
detection behavior survives the transfer. Our manuscript delineates
that boundary and localizes it at the distillation interface:

- **A mechanism-level finding, not just a negative verdict**: the
  distillation interface, not distillation itself, decides what
  transfers. Score-level distillation transfers almost none of the
  Chronos teacher's ranking behavior and yields no detectable quality
  advantage (the full-flow TimesFM extension adds an instructive split:
  its ranking does transfer at 0.70 and still yields none); the teacher's point forecasts as the training target
  recover most of the gap on development flows and are statistically
  equivalent to self-supervision on held-out sources. Four
  controls—shuffled-teacher, equal-context, anomaly-exposure, and a
  second teacher family—attribute the failure to the distillation
  interface rather than to student capacity, within the tested configurations.
- **A preregistered, source-blocked protocol** (frozen design with an
  append-only, hash-chained revision log; 29 flows from 22 public data
  collections; 1,270 preregistered runs plus 586 disclosed additions,
  1,856 runs in total; eleven arms) comparing score-, ranking-, forecast-, and
  self-supervised training of compact (~0.27M-parameter) students
  against the frozen Chronos-Bolt teacher, TimesFM, and classical
  references.
- **A deployment-cost axis aligned with the journal's scope**: measured
  on trained ONNX artifacts with optional INT8 quantization, students
  run 0.07–0.51 ms per window (p50)—roughly 200–1500× faster than the
  teacher—and quantized PatchMLP students fit sub-millisecond,
  sub-megabyte budgets at every tested channel count (1 to 248), with
  per-configuration quantization quality drift quantified. On-device
  validation on a Raspberry Pi 4B (single-thread ONNX Runtime, batch 1)
  replicates the protocol at the reference dimension: the students measure
  0.44-15.47 ms p95 on board, remaining 8-278x below the teacher desktop
  cost.

We believe this combination—a rigorous boundary delineation of a
widely assumed technique, plus concrete edge-deployment measurements—
fits the journal's interest in system performance, resource-constrained
computing, and dependable empirical evaluation.

This manuscript is original, has not been published previously, and is
not under consideration elsewhere. All authors have approved the
submission and declare no conflicts of interest. All benchmark data are
public; code, the preregistration with its hash chain, and per-run
results will be released with an archived DOI upon acceptance (review
complete artifact package accompanies the submission).

Thank you for your consideration.

Sincerely,
Yao Zhang (corresponding author)
Department of Artificial Intelligence, Baoding University
Baoding, China
Email: zhangyao@bdu.edu.cn
