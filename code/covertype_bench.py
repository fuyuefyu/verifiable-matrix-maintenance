#!/usr/bin/env python3
"""Fixed-reference Covertype row-update benchmark; requires adjacent exact_field_bench.py."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
  try:
    _count = int(os.environ.get(_name, '4'))
  except ValueError:
    _count = 4
  os.environ[_name] = str(min(4, max(1, _count)))
import argparse
import gc
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
try:
  from threadpoolctl import threadpool_info, threadpool_limits
except ImportError:
  threadpool_info = threadpool_limits = None
from exact_field_bench import Q, DEFAULT_SEED_IDS, Backend, Clock, canonical, freeze, fresh_vectors, matrix_digest, modadd, modmatmul, python_selftest, rng_for
FEATURES = 54
ARRAY_LIMIT = 2 ** 31 - 1
METHODS = ('recompute_check', 'incremental_check', 'cached_factorized_check', 'cached_factorized_no_check')
ROUND_PHASES = ('input_transfer', 'factor_difference', 'factor_construction', 'row_state_update', 'product', 'candidate_update', 'freeze', 'candidate_receive', 'randomness', 'challenge_transfer', 'verification', 'state_advance', 'rollback')

def read_manifest(path):
  path = Path(path).resolve()
  raw = path.read_bytes()
  obj = json.loads(raw)
  if obj.get('class_target_excluded') is not True:
    raise ValueError('Manifest must explicitly exclude the class target')
  streams = {}
  for item in obj['streams']:
    seed = int(item['seed'])
    if seed in streams or int(item['stream_id']) != seed:
      raise ValueError('Duplicate or inconsistent stream identity')
    streams[seed] = item
  return (path, obj, streams, hashlib.sha256(raw).hexdigest())

def load_stream(manifest_path, stream, n, needed):
  expected = {'X': (1024, 54), 'Y': (1024, 54), 'replacement': (8192, 54)}
  values, receipts = ({}, {})
  root = manifest_path.parent
  for name, shape in expected.items():
    desc = stream['arrays'][name]
    if tuple(desc['shape']) != shape or desc['dtype'] != '<i4':
      raise ValueError('Unexpected shape or dtype for ' + name)
    relative = Path(desc['path'])
    if relative.is_absolute():
      raise ValueError('Manifest array paths must be relative')
    source = (root / relative).resolve()
    if os.path.commonpath([str(root), str(source)]) != str(root):
      raise ValueError('Manifest array escapes its directory')
    size = shape[0] * shape[1] * 4
    if int(desc['byte_size']) != size or source.stat().st_size != size:
      raise ValueError('Binary byte_size mismatch: ' + name)
    raw = source.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != desc['binary_sha256']:
      raise ValueError('Binary SHA-256 mismatch: ' + name)
    arr = np.frombuffer(raw, dtype='<i4').reshape(shape)
    count = needed if name == 'replacement' else n
    if count > shape[0]:
      raise ValueError('Insufficient source rows; repetition is not permitted')
    value = np.remainder(arr[:count].astype(np.int64), Q).astype('<u4')
    value.flags.writeable = False
    values[name] = value
    receipts[name] = {key: desc[key] for key in ('path', 'shape', 'dtype', 'byte_size', 'binary_sha256', 'raw_records_sha256', 'source_row_start', 'source_row_stop_exclusive') if key in desc}
  return (values['X'], values['Y'], values['replacement'], receipts)

def indices_array(ctx, host):
  host = np.ascontiguousarray(host, dtype=np.int64)
  if ctx.gpu:
    ctx.h2d += int(host.nbytes)
    return ctx.xp.asarray(host)
  return host.copy()

def measured(ctx, profile_sync, phase_names, operation):
  ctx.sync()
  ctx.reset_peak()
  before = ctx.memory()
  tx0, rx0 = (ctx.h2d, ctx.d2h)
  clock = Clock(ctx, profile_sync)
  clock.ns = dict.fromkeys(phase_names, 0)
  start = time.perf_counter_ns()
  value = operation(clock)
  ctx.sync()
  total = time.perf_counter_ns() - start
  residual = total - sum(clock.ns.values())
  if residual < 0:
    raise AssertionError('Negative timing residual')
  clock.ns['orchestration'] = residual
  after = ctx.memory()
  if ctx.gpu and after['gpu_array_peak_bytes'] >= ARRAY_LIMIT:
    raise MemoryError('Live GPU array allocation exceeds the specified cap')
  receipt = {'total_ns': total, 'total_ms': total / 1000000.0, 'timing_mode': 'profile_sync' if profile_sync else 'primary_e2e', 'memory_before': before, 'memory_after': after, 'h2d_payload_bytes': ctx.h2d - tx0, 'd2h_payload_bytes': ctx.d2h - rx0}
  if ctx.gpu and (not profile_sync):
    prefix = 'host_observed_phase'
    receipt['phase_time_semantics'] = 'host_observed_calls_and_necessary_waits'
  else:
    prefix = 'component_phase'
    receipt['phase_time_semantics'] = 'synchronized_stage_wall_time' if ctx.gpu else 'cpu_stage_wall_time'
  receipt[prefix + '_ns'] = clock.ns
  receipt[prefix + '_ms'] = {key: value / 1000000.0 for key, value in clock.ns.items()}
  return (value, receipt)

def fixture_b(ctx, n, seed):
  """B=LU is fixture generation, excluded from production preprocessing."""
  rng = rng_for(seed, 45553)
  lower = np.tril(rng.integers(0, Q, (n, n), dtype=np.uint32))
  upper = np.triu(rng.integers(0, Q, (n, n), dtype=np.uint32))
  np.fill_diagonal(lower, 1)
  np.fill_diagonal(upper, 1)
  return canonical(ctx, modmatmul(ctx, ctx.array(lower), ctx.array(upper)))

def prepare_base(ctx, xh, yh, bh, profile_sync, order, deadline):

  def common(clock):
    x, y, b = clock.call('input_transfer', lambda: (ctx.array(xh), ctx.array(yh), ctx.array(bh)))
    a = clock.call('initial_A', lambda: modmatmul(ctx, x, y.T))
    return (x, y, b, a)
  deadline()
  arrays, common_receipt = measured(ctx, profile_sync, ('input_transfer', 'initial_A'), common)
  x, y, b, a = arrays
  base = {'X0': x, 'Y': y, 'B': b, 'A0': a, 'Y_host': yh, 'B_host': bh, 'common_setup': common_receipt, 'initialization_order': list(order)}
  for name in order:
    deadline()
    if name == 'uncached':
      c0, receipt = measured(ctx, profile_sync, ('initial_C',), lambda clock: clock.call('initial_C', lambda: modmatmul(ctx, a, b)))
      base['C0_uncached_host'] = canonical(ctx, c0)
      del c0
    else:

      def cached(clock):
        h = clock.call('initial_H', lambda: modmatmul(ctx, y.T, b))
        c0 = clock.call('initial_C', lambda: modmatmul(ctx, x, h))
        return (h, c0)
      (h, c0), receipt = measured(ctx, profile_sync, ('initial_H', 'initial_C'), cached)
      base['H_host'] = canonical(ctx, h)
      base['C0_cached_host'] = canonical(ctx, c0)
      del h, c0
    base[name + '_initialization'] = receipt
    deadline()
  if not np.array_equal(base['C0_uncached_host'], base['C0_cached_host']):
    raise AssertionError('Uncached and factorized initial products disagree')
  base['C0_equality_checked_outside_timer'] = True
  base['production_setup_ns'] = {name: common_receipt['total_ns'] + base[name + '_initialization']['total_ns'] for name in ('uncached', 'cached')}
  deadline()
  return base

def new_state(ctx, base, method):
  cached = method.startswith('cached_')
  state = {'X': base['X0'].copy(), 'A': base['A0'].copy(), 'C': ctx.array(base['C0_cached_host' if cached else 'C0_uncached_host']), 'H': ctx.array(base['H_host']) if cached else None}
  ctx.sync()
  return state

def row_check(ctx, candidate, previous, vt, b, idx, z):
  lhs = modmatmul(ctx, modadd(ctx, candidate, -previous), z)
  selected = modmatmul(ctx, vt, modmatmul(ctx, b, z))
  rhs = ctx.xp.zeros(lhs.shape, dtype=ctx.xp.float64)
  rhs[idx] = selected
  return ctx.truth(ctx.xp.all(lhs == rhs))

def rollback(state, undo):
  if undo:
    state['A'][undo['idx']] = undo['A']
    state['X'][undo['idx']] = undo['X']
    state['C'] = undo['C']

def execute_round(ctx, state, base, new_host, idx_host, t, method, profile_sync, context):
  if method not in METHODS:
    raise ValueError('Unknown method')
  undo = {}

  def operation(clock):
    new_rows, idx = clock.call('input_transfer', lambda: (ctx.array(new_host), indices_array(ctx, idx_host)))

    def difference():
      old_x = state['X'][idx]
      return (old_x, modadd(ctx, new_rows, -old_x))
    old_x, d = clock.call('factor_difference', difference)
    vt = clock.call('factor_construction', lambda: modmatmul(ctx, d, base['Y'].T))

    def update_rows():
      old_a = state['A'][idx]
      undo.update(idx=idx, A=old_a, X=old_x, C=state['C'])
      state['A'][idx] = modadd(ctx, old_a, vt)
      state['X'][idx] = new_rows
    clock.call('row_state_update', update_rows)

    def product():
      if method == 'recompute_check':
        return modmatmul(ctx, state['A'], base['B'])
      if method == 'incremental_check':
        return modmatmul(ctx, vt, base['B'])
      return modmatmul(ctx, d, state['H'])
    result = clock.call('product', product)

    def candidate():
      if method == 'recompute_check':
        return result
      full = undo['C'].copy()
      full[idx] = modadd(ctx, undo['C'][idx], result)
      return full
    proposed = clock.call('candidate_update', candidate)
    frozen, digest = clock.call('freeze', lambda: freeze(ctx, proposed, context))
    received = clock.call('candidate_receive', lambda: ctx.array(frozen))
    zh = None
    accepted = True
    if method != 'cached_factorized_no_check':
      zh = clock.call('randomness', lambda: fresh_vectors(state['A'].shape[0], t))
      z = clock.call('challenge_transfer', lambda: ctx.array(zh))
      accepted = clock.call('verification', lambda: row_check(ctx, received, undo['C'], vt, base['B'], idx, z))
    if accepted:

      def advance():
        state['C'] = received
      clock.call('state_advance', advance)
    else:
      clock.call('rollback', lambda: rollback(state, undo))
    return (frozen, digest, zh, accepted)
  try:
    result, row = measured(ctx, profile_sync, ROUND_PHASES, operation)
  except Exception:
    rollback(state, undo)
    ctx.sync()
    raise
  frozen, digest, zh, accepted = result
  retained = sum((int(state[key].nbytes) for key in ('X', 'A', 'C')))
  retained += int(base['Y'].nbytes + base['B'].nbytes)
  h_bytes = 0 if state['H'] is None else int(state['H'].nbytes)
  row.update({'method': method, 'accepted': accepted, 'candidate_context': context, 'candidate_sha256': digest, 'challenge_sha256': None if zh is None else matrix_digest(zh), 'resident_state_array_bytes': retained + h_bytes, 'retained_H_array_bytes': h_bytes, 'retained_H_canonical_bytes': h_bytes // 2, 'device_reset_snapshot_bytes': int(base['X0'].nbytes + base['A0'].nbytes), 'host_reset_snapshot_bytes': int(base['C0_uncached_host'].nbytes + base['C0_cached_host'].nbytes + base['H_host'].nbytes)})
  return (row, frozen, zh, undo)

def validate_round(ctx, state, base, expected_x, frozen, idx_host, undo, seed, round_id):
  oracle = modmatmul(ctx, state['A'], base['B'])
  full_ok = ctx.truth(ctx.xp.all(oracle == state['C']))
  changed_x = np.array(ctx.host(state['X'][undo['idx']]), dtype=np.uint32)
  spot_ok = np.array_equal(changed_x, expected_x[idx_host])
  n = expected_x.shape[0]
  rows = [int(idx_host[0]), int(idx_host[-1]), (int(idx_host[-1]) + 1) % n, (seed + 53 * round_id) % n]
  cols = [(seed + 17 * round_id + 23 * k) % n for k in range(4)]
  for i, j in zip(rows, cols):
    xr = np.array(ctx.host(state['X'][i, :]), dtype=np.uint32)
    ar = np.array(ctx.host(state['A'][i, :]), dtype=np.uint32)
    spot_ok &= np.array_equal(xr, expected_x[i])
    a_want = sum((int(expected_x[i, k]) * int(base['Y_host'][j, k]) for k in range(FEATURES))) % Q
    c_want = sum((int(ar[k]) * int(base['B_host'][k, j]) for k in range(n))) % Q
    spot_ok &= int(ar[j]) == a_want and int(frozen[i, j]) == c_want
  ctx.sync()
  return (bool(full_ok), bool(spot_ok))

def python_product(left, right):
  return np.array([[sum((int(left[i, k]) * int(right[k, j]) for k in range(left.shape[1]))) % Q for j in range(right.shape[1])] for i in range(left.shape[0])], dtype='<u4')

def small_full_check(ctx, profile_sync, deadline):
  rng = rng_for(7654321, 20817)
  xh = rng.integers(0, Q, (4, FEATURES), dtype=np.uint32)
  yh = rng.integers(0, Q, (4, FEATURES), dtype=np.uint32)
  replacements = rng.integers(0, Q, (4, FEATURES), dtype=np.uint32)
  bh = fixture_b(ctx, 4, 7654321)
  base = prepare_base(ctx, xh, yh, bh, profile_sync, ('uncached', 'cached'), deadline)
  initial = python_product(python_product(xh, yh.T), bh)
  if not np.array_equal(initial, base['C0_uncached_host']):
    raise AssertionError('Small full initial bigint check failed')
  for method in METHODS:
    state, expected = (new_state(ctx, base, method), xh.copy())
    for round_id in range(2):
      deadline()
      idx = np.arange(round_id * 2, round_id * 2 + 2, dtype=np.int64)
      fresh = replacements[round_id * 2:round_id * 2 + 2]
      context = {'selftest': True, 'seed': 7654321, 'repeat': 0, 'round': round_id, 'n': 4, 'b': 2, 't': 2}
      row, frozen, _, _ = execute_round(ctx, state, base, fresh, idx, 2, method, profile_sync, context)
      expected[idx] = fresh
      expected_a = python_product(expected, yh.T)
      expected_c = python_product(expected_a, bh)
      if not row['accepted'] or not np.array_equal(canonical(ctx, state['X']), expected) or (not np.array_equal(canonical(ctx, state['A']), expected_a)) or (not np.array_equal(frozen, expected_c)):
        raise AssertionError('Small full Python-bigint row-update check failed')
      deadline()

def warmup(ctx, args, seed, deadline):
  rng = rng_for(seed, 2577)
  xh = rng.integers(0, Q, (args.n, FEATURES), dtype=np.uint32)
  yh = rng.integers(0, Q, (args.n, FEATURES), dtype=np.uint32)
  fresh = rng.integers(0, Q, (args.batch_size, FEATURES), dtype=np.uint32)
  base = prepare_base(ctx, xh, yh, fixture_b(ctx, args.n, seed), args.profile_sync, ('uncached', 'cached'), deadline)
  idx = np.arange(args.batch_size, dtype=np.int64)
  for method in METHODS:
    deadline()
    state, expected = (new_state(ctx, base, method), xh.copy())
    context = {'warmup': True, 'seed': seed, 'repeat': 0, 'round': 0, 'n': args.n, 'b': args.batch_size, 't': args.t}
    row, frozen, _, undo = execute_round(ctx, state, base, fresh, idx, args.t, method, args.profile_sync, context)
    expected[idx] = fresh
    good = validate_round(ctx, state, base, expected, frozen, idx, undo, seed, 0)
    if not row['accepted'] or not all(good):
      raise AssertionError('Independent-state warmup failed')
    deadline()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--manifest', required=True)
  parser.add_argument('--backend', choices=('cpu', 'gpu'), required=True)
  parser.add_argument('--device', type=int, default=0)
  parser.add_argument('--n', type=int, choices=(256, 512, 1024), required=True)
  parser.add_argument('--batch-size', type=int, choices=(2, 8, 32), required=True)
  parser.add_argument('--t', type=int, default=2)
  parser.add_argument('--rounds', type=int, default=64)
  parser.add_argument('--seeds', type=int, default=5)
  parser.add_argument('--repeats', type=int, default=3)
  parser.add_argument('--seed-ids', default=DEFAULT_SEED_IDS)
  parser.add_argument('--seed-offset', type=int, default=0)
  parser.add_argument('--profile-sync', action='store_true')
  parser.add_argument('--max-seconds', type=float, default=6600)
  parser.add_argument('--out', required=True)
  args = parser.parse_args()
  if min(args.t, args.rounds, args.seeds, args.repeats) < 1 or args.seed_offset < 0:
    parser.error('Counts must be positive and seed-offset nonnegative')
  if not np.isfinite(args.max_seconds) or args.max_seconds <= 0:
    parser.error('max-seconds must be finite and positive')
  if args.rounds * args.batch_size > 8192:
    parser.error('Fresh replacement rows exhausted; no recycling is allowed')
  try:
    seed_ids = [int(x.strip()) for x in args.seed_ids.split(',')]
  except ValueError:
    parser.error('Invalid seed list')
  if not seed_ids or min(seed_ids) < 0 or len(set(seed_ids)) != len(seed_ids):
    parser.error('Seed list must be distinct and nonnegative')
  end = args.seed_offset + args.seeds
  if end > len(seed_ids):
    parser.error('Requested seed slice exceeds the seed list')
  selected = seed_ids[args.seed_offset:end]
  array_bound = 96 * 8 * args.n ** 2 + 32 * 8 * FEATURES * args.n + 64 * 8192 * FEATURES
  if array_bound >= ARRAY_LIMIT:
    parser.error('Conservative live-array bound exceeds 2 GiB')
  Path(args.out).resolve().parent.mkdir(parents=True, exist_ok=True)
  started, completed, location = (time.monotonic(), 0, {'stage': 'startup'})
  with open(args.out, 'x', encoding='utf-8', buffering=1) as output:

    def emit(obj):
      output.write(json.dumps(obj, separators=(',', ':'), allow_nan=False) + '\n')

    def deadline():
      if time.monotonic() - started >= args.max_seconds:
        raise TimeoutError('Cooperative cell budget includes setup and all oracle work')
    try:
      if threadpool_limits is None:
        raise RuntimeError('threadpoolctl is required to enforce the 4-thread limit')
      ctx = Backend(args.backend, args.device)
      limiter = threadpool_limits(limits=4)
      pools = threadpool_info()
      if any((int(p.get('num_threads', 0)) > 4 for p in pools)):
        raise RuntimeError('A detected CPU compute pool exceeds four threads')
      if ctx.gpu:
        existing = ctx.pool.get_limit()
        ctx.pool.set_limit(size=min(existing, ARRAY_LIMIT) if existing else ARRAY_LIMIT)
      mp, manifest, streams, manifest_sha = read_manifest(args.manifest)
      if any((seed not in streams for seed in selected)):
        raise ValueError('Requested seed has no manifest stream')
      warm_seed = max(seed_ids + [987654321]) + 1
      emit({'kind': 'metadata', 'version': 'covertype-row-bench-v1', 'args': vars(args), 'q': Q, 'features': FEATURES, 'selected_seed_ids': selected, 'manifest_sha256': manifest_sha, 'manifest_indexing': manifest.get('indexing'), 'selected_actual_rows': manifest.get('selected_actual_rows'), 'hostname': platform.node(), 'device_name': ctx.name, 'numpy_version': np.__version__, 'backend_version': ctx.xp.__version__, 'threadpool_info': pools, 'cpu_compute_thread_limit': 4, 'conservative_array_bound_bytes': array_bound, 'gpu_array_cap_bytes': ARRAY_LIMIT, 'timing_mode': 'profile_sync' if args.profile_sync else 'primary_e2e', 'component_times_available': bool(not ctx.gpu or args.profile_sync), 'workload': 'A=X Y^T with fixed reference Y; true source-row replacements', 'theory_scope': 'application-derived performance; no uniform-factor bound applied', 'oracle_scope': 'same-backend full AB; small full bigint and four per-round spots', 'setup_reuse': 'one measured initialization per seed, reused across repetitions', 'memory_scope': 'GPU live-array peak; RSS HWM process-lifetime', 'deadline_scope': 'cooperative operation boundaries; external hard stop required'})
      deadline()
      python_selftest(ctx, args.n)
      small_full_check(ctx, args.profile_sync, deadline)
      warm_start = time.perf_counter_ns()
      warmup(ctx, args, warm_seed, deadline)
      gc.collect()
      emit({'kind': 'warmup', 'wall_ns': time.perf_counter_ns() - warm_start, 'seed': warm_seed})
      for si, seed in enumerate(selected):
        deadline()
        location = {'stage': 'input', 'seed': seed}
        load_start = time.perf_counter_ns()
        xh, yh, replacements, files = load_stream(mp, streams[seed], args.n, args.rounds * args.batch_size)
        load_ns = time.perf_counter_ns() - load_start
        fixture_start = time.perf_counter_ns()
        bh = fixture_b(ctx, args.n, seed)
        fixture_ns = time.perf_counter_ns() - fixture_start
        deadline()
        global_index = args.seed_offset + si
        init_order = ('uncached', 'cached') if global_index % 2 == 0 else ('cached', 'uncached')
        base = prepare_base(ctx, xh, yh, bh, args.profile_sync, init_order, deadline)
        emit({'kind': 'setup', 'seed': seed, 'input_files': files, 'input_load_ns_outside_production': load_ns, 'B_fixture_ns_outside_production': fixture_ns, 'B_fixture_sha256': matrix_digest(bh), 'common_setup': base['common_setup'], 'uncached_initialization': base['uncached_initialization'], 'cached_initialization': base['cached_initialization'], 'initialization_order': base['initialization_order'], 'production_setup_ns': base['production_setup_ns'], 'C0_equality_checked_outside_timer': True})
        for repeat in range(args.repeats):
          shift = (global_index * args.repeats + repeat) % len(METHODS)
          order = METHODS[shift:] + METHODS[:shift]
          hashes = {}
          for position, method in enumerate(order):
            reset_start = time.perf_counter_ns()
            state = new_state(ctx, base, method)
            expected_x = xh.copy()
            reset_ns = time.perf_counter_ns() - reset_start
            setup_ns = base['production_setup_ns']['cached' if method.startswith('cached_') else 'uncached']
            cumulative = 0
            emit({'kind': 'trial_reset', 'seed': seed, 'repeat': repeat, 'method': method, 'order_position': position, 'reset_ns_outside_production': reset_ns})
            for round_id in range(args.rounds):
              deadline()
              b = args.batch_size
              idx = (round_id * b + np.arange(b, dtype=np.int64)) % args.n
              fresh = replacements[round_id * b:(round_id + 1) * b]
              context = {'seed': seed, 'repeat': repeat, 'round': round_id, 'n': args.n, 'b': b, 't': args.t}
              location = dict(context, stage='round', method=method)
              row, frozen, zh, undo = execute_round(ctx, state, base, fresh, idx, args.t, method, args.profile_sync, context)
              row.update(dict(context, kind='round', order_position=position, factor_width=b, source_replacement_start=round_id * b))
              if not row['accepted']:
                row.update(trial_valid=False, failed_check_vectors=None if zh is None else zh.tolist())
                emit(row)
                raise AssertionError('Verifier rejected an intended exact candidate')
              expected_x[idx] = fresh
              diagnostic_start = time.perf_counter_ns()
              try:
                full_ok, spot_ok = validate_round(ctx, state, base, expected_x, frozen, idx, undo, seed, round_id)
                hash_ok = round_id not in hashes or hashes[round_id] == row['candidate_sha256']
              except Exception:
                rollback(state, undo)
                ctx.sync()
                raise
              valid = bool(full_ok and spot_ok and hash_ok)
              row.update({'validation_ns_outside_timer': time.perf_counter_ns() - diagnostic_start, 'full_AB_match': full_ok, 'python_bigint_spots_match': spot_ok, 'cross_method_candidate_match': hash_ok, 'trial_valid': valid})
              if not valid:
                rollback(state, undo)
                ctx.sync()
                row['diagnostic_rollback'] = True
                emit(row)
                raise AssertionError('State, bigint or cross-method validation failed')
              hashes[round_id] = row['candidate_sha256']
              cumulative += row['total_ns']
              row.update(production_setup_ns=setup_ns, cumulative_update_ns=cumulative, cumulative_production_ns=setup_ns + cumulative)
              emit(row)
              completed += 1
              del frozen, zh, undo
              deadline()
            del state
        del base, xh, yh, replacements, bh
        gc.collect()
        deadline()
      deadline()
      emit({'kind': 'completed', 'round_records': completed, 'wall_seconds': time.monotonic() - started})
    except Exception as error:
      emit({'kind': 'failed', 'error_type': type(error).__name__, 'error': str(error), 'location': location, 'completed_round_records': completed, 'wall_seconds': time.monotonic() - started})
      raise
  return 0
if __name__ == '__main__':
  raise SystemExit(main())
