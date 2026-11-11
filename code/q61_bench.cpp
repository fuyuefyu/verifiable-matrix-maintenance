// C++17/g++ exact q61 CPU subset; one (n,r) per process. No benchmark run yet.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Word = std::uint64_t;
using Wide = unsigned __int128;
using Clock = std::chrono::steady_clock;
constexpr Word Q = (Word(1) << 61) - 1;
constexpr Word mul(Word a, Word b) { return Word(Wide(a) * b % Q); }
constexpr Word add(Word a, Word b) { return a + b >= Q ? a + b - Q : a + b; }
constexpr Word sub(Word a, Word b) { return a >= b ? a - b : Q - (b - a); }
static_assert(mul(Q - 1, Q - 1) == 1, "q61 boundary product");
static_assert(mul(Q - 1, Q - 2) == 2, "q61 boundary product");
static_assert(mul(0, Q - 1) == 0 && add(Q - 1, 1) == 0, "q61 boundaries");

Word elapsed(Clock::time_point s) {
 return Word(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - s).count());
}
struct Deadline : std::runtime_error { Deadline() : std::runtime_error("max-seconds exceeded") {} };
struct Limit {
 Clock::time_point start;
 Word seconds;
 void check() const {
  if (Clock::now() - start >= std::chrono::seconds(seconds)) throw Deadline();
 }
};
template<class F> void timed(Word &ns, F fn) {
 const auto s = Clock::now();
 try { fn(); } catch (...) { ns += elapsed(s); throw; }
 ns += elapsed(s);
}

struct Matrix {
 int rows = 0, cols = 0;
 std::vector<Word> a;
 Matrix() = default;
 Matrix(int r, int c) : rows(r), cols(c), a(std::size_t(r) * c, 0) {}
 Word &at(int i, int j) { return a[std::size_t(i) * cols + j]; }
 Word at(int i, int j) const { return a[std::size_t(i) * cols + j]; }
 bool operator==(const Matrix &b) const { return rows == b.rows && cols == b.cols && a == b.a; }
};
Matrix mm(const Matrix &a, const Matrix &b, const Limit &limit) {
 if (a.cols != b.rows) throw std::runtime_error("matrix shape mismatch");
 Matrix c(a.rows, b.cols);
 for (int i = 0; i < a.rows; ++i) {
  limit.check();
  for (int k = 0; k < a.cols; ++k) {
   const Word v = a.at(i, k);
   if (!v) continue;
   for (int j = 0; j < b.cols; ++j)
    c.at(i, j) = add(c.at(i, j), mul(v, b.at(k, j)));
  }
 }
 return c;
}
Matrix transpose(const Matrix &a, const Limit &limit) {
 Matrix b(a.cols, a.rows);
 for (int i = 0; i < a.rows; ++i) {
  limit.check();
  for (int j = 0; j < a.cols; ++j) b.at(j, i) = a.at(i, j);
 }
 return b;
}
void plus_into(Matrix &a, const Matrix &b, const Limit &limit) {
 if (a.rows != b.rows || a.cols != b.cols) throw std::runtime_error("addition shape mismatch");
 for (int i = 0; i < a.rows; ++i) {
  limit.check();
  for (int j = 0; j < a.cols; ++j) a.at(i, j) = add(a.at(i, j), b.at(i, j));
 }
}
Word input_word(std::mt19937_64 &rng) {
 Word x;
 do { x = rng() & Q; } while (x == Q);
 return x;
}
Matrix random_matrix(int n, int m, std::mt19937_64 &rng, const Limit &limit) {
 Matrix a(n, m);
 for (int i = 0; i < n; ++i) {
  limit.check();
  for (int j = 0; j < m; ++j) a.at(i, j) = input_word(rng);
 }
 return a;
}
Matrix unit_lu(int n, std::mt19937_64 &rng, const Limit &limit) {
 Matrix l(n, n), u(n, n);
 for (int i = 0; i < n; ++i) {
  limit.check();
  l.at(i, i) = u.at(i, i) = 1;
  for (int j = 0; j < i; ++j) l.at(i, j) = input_word(rng);
  for (int j = i + 1; j < n; ++j) u.at(i, j) = input_word(rng);
 }
 return mm(l, u, limit); // det(L)=det(U)=1, hence B is invertible.
}

// Called after const delivery. A new stream avoids old-candidate prefetch.
Matrix fresh_vectors(int t, int n, const Limit &limit) {
 limit.check();
 std::ifstream entropy("/dev/urandom", std::ios::binary);
  if (!entropy) throw std::runtime_error("/dev/urandom unavailable; no fallback");
 Matrix z(t, n);
 for (int i = 0; i < t; ++i) {
  for (int j = 0; j < n; ++j) {
   limit.check();
   Word x;
   do {
    entropy.read(reinterpret_cast<char *>(&x), sizeof x);
    if (!entropy) throw std::runtime_error("/dev/urandom read failed");
    x &= Q; // Uniform in [0,Q]; reject Q to obtain [0,Q-1].
   } while (x == Q);
   z.at(i, j) = x;
  }
 }
 return z;
}
bool verify(const Matrix &candidate, const Matrix &old_c, const Matrix &b,
   const Matrix &u, const Matrix &v, const Matrix &zs, const Limit &limit) {
 bool ok = true;
 const int n = b.rows, r = u.cols;
 for (int h = 0; h < zs.rows; ++h) {
  std::vector<Word> bz(n, 0), vbz(r, 0);
  for (int i = 0; i < n; ++i) {
   limit.check();
   for (int j = 0; j < n; ++j) bz[i] = add(bz[i], mul(b.at(i, j), zs.at(h, j)));
  }
  for (int k = 0; k < r; ++k) {
   limit.check();
   for (int i = 0; i < n; ++i) vbz[k] = add(vbz[k], mul(v.at(i, k), bz[i]));
  }
  for (int i = 0; i < n; ++i) {
   limit.check();
   Word lhs = 0, rhs = 0;
   for (int j = 0; j < n; ++j)
    lhs = add(lhs, mul(sub(candidate.at(i, j), old_c.at(i, j)), zs.at(h, j)));
   for (int k = 0; k < r; ++k) rhs = add(rhs, mul(u.at(i, k), vbz[k]));
   if (lhs != rhs) ok = false; // Execute all t checks, including on failure.
  }
 }
 return ok;
}

struct Config {
 int n = 0, r = 0, t = 2, repeats = 3, rounds = 4;
 Word max_seconds = 900;
 std::vector<Word> seeds{101, 211, 307, 401, 503};
 std::string out = "q61_results", trace_out;
};
Word number(const std::string &s) {
 if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
  throw std::runtime_error("expected unsigned integer: " + s);
 return std::stoull(s);
}
int bounded(const std::string &s, int high) {
 const Word x = number(s);
 if (!x || x > Word(high)) throw std::runtime_error("argument outside supported range: " + s);
 return int(x);
}
Config parse(int argc, char **argv) {
 Config c;
 for (int i = 1; i < argc; ++i) {
  const std::string k = argv[i];
  if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
  const std::string v = argv[++i];
  if (k == "--n") c.n = bounded(v, 256);
  else if (k == "--r") c.r = bounded(v, 256);
  else if (k == "--t") c.t = bounded(v, 64);
  else if (k == "--reps" || k == "--repeats") c.repeats = bounded(v, 64);
  else if (k == "--rounds") c.rounds = bounded(v, 64);
  else if (k == "--max-seconds") c.max_seconds = Word(bounded(v, 43200));
  else if (k == "--out") c.out = v;
  else if (k == "--trace-out") c.trace_out = v;
  else if (k == "--seeds") {
   c.seeds.clear();
   std::stringstream ss(v);
   std::string item;
   while (std::getline(ss, item, ',')) c.seeds.push_back(number(item));
   if (c.seeds.empty() || c.seeds.size() > 64 || v.back() == ',')
    throw std::runtime_error("invalid seed list");
   auto ids = c.seeds;
   std::sort(ids.begin(), ids.end());
   if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
    throw std::runtime_error("duplicate seed IDs");
  } else throw std::runtime_error("unknown option: " + k);
 }
 if (!c.n || !c.r || c.r > c.n) throw std::runtime_error("require --n and --r with 1<=r<=n<=256");
 if (c.out.empty()) throw std::runtime_error("empty output prefix");
 if (!c.trace_out.empty() && (c.n > 16 || c.r > 8 || c.rounds > 2 || c.repeats != 1 || c.seeds.size() != 1))
  throw std::runtime_error("trace requires n<=16,r<=8,rounds<=2,one seed and repeat");
 return c;
}
std::string host() {
 std::ifstream f("/etc/hostname");
 std::string s;
 if (std::getline(f, s) && !s.empty()) return s;
 const char *p = std::getenv("HOSTNAME");
 return p ? p : "unknown";
}
std::string jq(const std::string &s) {
 std::ostringstream o;
 o << '"';
 const char *hex = "0123456789abcdef";
 for (unsigned char c : s) {
  if (c == '"' || c == '\\') o << '\\' << char(c);
  else if (c < 32) o << "\\u00" << hex[c >> 4] << hex[c & 15];
  else o << char(c);
 }
 o << '"';
 return o.str();
}
std::string cq(const std::string &s) {
 std::string t = "\"";
 for (char c : s) { if (c == '"') t += '"'; t += c; }
 return t + '"';
}
void new_file(std::ofstream &f, const std::string &path) {
 std::ifstream old(path);
 if (old.good()) throw std::runtime_error("refusing to overwrite " + path);
 f.open(path, std::ios::out);
 if (!f) throw std::runtime_error("cannot create " + path);
 f.exceptions(std::ios::badbit | std::ios::failbit);
}
const std::array<std::string, 3> METHODS{"recompute_check", "incremental_check", "incremental_no_check"};
struct Row {
 Word seed = 0, total_ns = 0, oracle_ns = 0, setup_ns = 0;
 int repeat = 0, round = 0, method = 0;
 int check_pass = -1, full_ab = -1, common_a = -1, state_match = -1;
 bool frozen = false, fresh = false, check_performed = false, advanced = false;
 bool accepted = false, oracle_checked = false, trial_valid = false;
 std::string status = "incomplete", phase = "not_started";
};
std::string jb(bool v) { return v ? "true" : "false"; }
std::string flag(int v) { return v < 0 ? "null" : jb(v != 0); }
struct Output {
 const Config &c;
 std::string hostname;
 std::ofstream csv, json;
 Word records = 0, valid = 0;
 Output(const Config &cfg) : c(cfg), hostname(host()) {
  new_file(csv, c.out + ".csv"); new_file(json, c.out + ".jsonl");
  csv << "seed,repeat,round,method,n,r,t,q,hostname,timing_mode,total_ns,setup_ns,oracle_ns,accepted,trial_valid,full_AB_match,oracle_checked,common_A_match,state_match,check_performed,check_pass,advanced,candidate_frozen,fresh_vectors,status,phase\n";
  json << "{\"kind\":\"metadata\",\"version\":\"q61-field-bench-v1\",\"args\":{\"backend\":\"cpu\",\"device\":0,\"n\":" << c.n
   << ",\"r\":" << c.r << ",\"t\":" << c.t << ",\"rounds\":" << c.rounds << ",\"seeds\":" << c.seeds.size()
   << ",\"repeats\":" << c.repeats << ",\"max_seconds\":" << c.max_seconds << "},\"q\":" << jq(std::to_string(Q))
   << ",\"hostname\":" << jq(hostname) << ",\"selected_seed_ids\":[";
  for (std::size_t i = 0; i < c.seeds.size(); ++i) { if (i) json << ','; json << c.seeds[i]; }
  json << "],\"timing_mode\":\"primary_e2e\",\"threads\":1,\"implementation\":\"cpp17_uint128_exact_cpu_subset\",\"candidate_boundary\":\"deep_immutable_copy_before_urandom\",\"oracle_backend\":\"same_cpp_integer_full_AB\"}\n";
  csv.flush(); json.flush();
 }
 void setup(Word seed, int repeat, Word ns, const std::string &status = "complete") {
  json << "{\"kind\":\"setup\",\"seed\":" << seed << ",\"repeat\":" << repeat << ",\"setup_ns\":" << ns
   << ",\"n\":" << c.n << ",\"r\":" << c.r << ",\"B_construction\":\"unit_L_times_unit_U\",\"status\":" << jq(status) << "}\n";
  json.flush();
 }
 void emit(const Row &x) {
  ++records; if (x.trial_valid) ++valid;
  csv << x.seed << ',' << x.repeat << ',' << x.round << ',' << METHODS[x.method] << ',' << c.n << ',' << c.r << ',' << c.t << ',' << Q << ','
   << cq(hostname) << ",primary_e2e," << x.total_ns << ',' << x.setup_ns << ',' << x.oracle_ns << ',' << x.accepted << ',' << x.trial_valid << ','
   << x.full_ab << ',' << x.oracle_checked << ',' << x.common_a << ',' << x.state_match << ',' << x.check_performed << ',' << x.check_pass << ','
   << x.advanced << ',' << x.frozen << ',' << x.fresh << ',' << cq(x.status) << ',' << cq(x.phase) << '\n';
  json << "{\"kind\":\"round\",\"seed\":" << x.seed << ",\"repeat\":" << x.repeat << ",\"round\":" << x.round
   << ",\"method\":" << jq(METHODS[x.method]) << ",\"accepted\":" << jb(x.accepted) << ",\"trial_valid\":" << jb(x.trial_valid)
   << ",\"full_AB_match\":" << flag(x.full_ab) << ",\"total_ns\":" << x.total_ns << ",\"setup_ns\":" << x.setup_ns << ",\"oracle_ns\":" << x.oracle_ns
   << ",\"timing_mode\":\"primary_e2e\",\"n\":" << c.n << ",\"r\":" << c.r << ",\"t\":" << c.t
   << ",\"q\":" << jq(std::to_string(Q)) << ",\"hostname\":" << jq(hostname) << ",\"oracle_checked\":" << jb(x.oracle_checked)
   << ",\"common_A_match\":" << flag(x.common_a) << ",\"state_match\":" << flag(x.state_match) << ",\"check_performed\":" << jb(x.check_performed)
   << ",\"check_pass\":" << flag(x.check_pass) << ",\"advanced\":" << jb(x.advanced) << ",\"candidate_frozen\":" << jb(x.frozen)
   << ",\"fresh_vectors\":" << jb(x.fresh) << ",\"status\":" << jq(x.status) << ",\"phase\":" << jq(x.phase) << "}\n";
  csv.flush(); json.flush();
 }
 void finish(const std::string &status, const std::string &reason, Word ns) {
  const Word expected = Word(c.seeds.size()) * c.repeats * c.rounds * 3;
  json << "{\"kind\":" << jq(status == "completed" ? "completed" : "failed") << ",\"status\":" << jq(status)
   << ",\"reason\":" << jq(reason) << ",\"round_records\":" << records << ",\"valid_round_records\":" << valid
   << ",\"expected_round_records\":" << expected << ",\"wall_ns\":" << ns << "}\n";
  json.flush();
 }
};

void matrix_json(std::ostream &o, const Matrix &a) {
 o << "{\"rows\":" << a.rows << ",\"cols\":" << a.cols << ",\"data\":[";
 for (std::size_t i = 0; i < a.a.size(); ++i) { if (i) o << ','; o << jq(std::to_string(a.a[i])); }
 o << "]}"; // q61 values are decimal strings.
}
struct Trace {
 std::ofstream f;
 bool enabled = false, first = true;
 explicit Trace(const Config &c) {
  if (c.trace_out.empty()) return;
  new_file(f, c.trace_out); enabled = true;
  f << "{\"version\":\"q61-pilot-trace-v1\",\"q\":" << jq(std::to_string(Q)) << ",\"events\":[\n";
 }
 void sep() { if (!first) f << ",\n"; first = false; }
 void setup(Word seed, int repeat, const Matrix &a, const Matrix &b, const Matrix &c) {
  if (!enabled) return;
  sep(); f << "{\"kind\":\"setup\",\"seed\":" << seed << ",\"repeat\":" << repeat << ",\"A0\":"; matrix_json(f, a);
  f << ",\"B\":"; matrix_json(f, b); f << ",\"C0\":"; matrix_json(f, c); f << '}'; f.flush();
 }
 void emit(const Row &x, const Matrix &u, const Matrix &v, const Matrix &a, const Matrix &c,
   const Matrix &zs, const Matrix &post_a, const Matrix &post_c) {
  if (!enabled) return;
  sep(); f << "{\"kind\":\"round\",\"seed\":" << x.seed << ",\"repeat\":" << x.repeat << ",\"round\":" << x.round
    << ",\"method\":" << jq(METHODS[x.method]) << ",\"accepted\":" << jb(x.accepted) << ",\"advanced\":" << jb(x.advanced)
    << ",\"check_performed\":" << jb(x.check_performed) << ",\"check_pass\":" << flag(x.check_pass) << ",\"U\":";
  matrix_json(f, u); f << ",\"V\":"; matrix_json(f, v); f << ",\"A_attempted\":"; matrix_json(f, a);
  f << ",\"C_delivered\":"; matrix_json(f, c); f << ",\"check_vectors\":"; matrix_json(f, zs);
  f << ",\"post_A\":"; matrix_json(f, post_a); f << ",\"post_C\":"; matrix_json(f, post_c); f << '}'; f.flush();
 }
 void finish(const std::string &status) {
  if (enabled) { f << "\n],\"status\":" << jq(status) << "}\n"; f.flush(); enabled = false; }
 }
};

struct State { Matrix a, c; };
struct Update { Matrix u, v; };
Matrix oracle_update(const Matrix &old_a, const Matrix &u, const Matrix &v, const Limit &limit) {
 Matrix a = old_a;
 for (int i = 0; i < a.rows; ++i) {
  limit.check();
  for (int j = 0; j < a.cols; ++j) {
   Word d = 0;
   for (int k = 0; k < u.cols; ++k) d = add(d, mul(u.at(i, k), v.at(j, k)));
   a.at(i, j) = add(a.at(i, j), d);
  }
 }
 return a;
}
void run_round(State &state, const Matrix &b, const Update &update, const Config &cfg,
   const Limit &limit, Row &row, Trace &trace) {
 State oracle_before;
 row.phase = "oracle_snapshot";
 timed(row.oracle_ns, [&] { limit.check(); oracle_before = state; });
 Matrix a, old_c, bc, u, v, vt, candidate, zs;
 std::unique_ptr<const Matrix> delivered;
 timed(row.total_ns, [&] {
  row.phase = "data_copies"; limit.check();
  a = state.a; old_c = state.c; bc = b; u = update.u; v = update.v;
  vt = transpose(v, limit);
  row.phase = "common_A_update";
  plus_into(a, mm(u, vt, limit), limit);
  row.phase = "operator";
  if (row.method == 0) candidate = mm(a, bc, limit);
  else {
   const Matrix vtb = mm(vt, bc, limit);
   const Matrix delta = mm(u, vtb, limit);
   candidate = old_c; plus_into(candidate, delta, limit);
  }
  row.phase = "immutable_delivery"; limit.check();
  delivered = std::make_unique<const Matrix>(candidate); // Deep copy, no mutable alias.
  row.frozen = true;
  if (row.method != 2) {
   row.phase = "fresh_vectors";
   zs = fresh_vectors(cfg.t, cfg.n, limit); row.fresh = true;
   row.phase = "verification"; row.check_performed = true;
   row.check_pass = verify(*delivered, old_c, bc, u, v, zs, limit) ? 1 : 0;
  }
  row.phase = "state_advance"; limit.check();
  row.accepted = row.method == 2 || row.check_pass == 1;
  if (row.accepted) { State next{a, *delivered}; std::swap(state, next); row.advanced = true; }
 });
 row.phase = "full_integer_oracle";
 timed(row.oracle_ns, [&] {
  const Matrix expected_a = oracle_update(oracle_before.a, update.u, update.v, limit);
  const Matrix expected_c = mm(expected_a, b, limit);
  row.common_a = a == expected_a ? 1 : 0;
  row.full_ab = *delivered == expected_c ? 1 : 0;
  row.state_match = row.advanced ? (state.a == a && state.c == *delivered) :
          (state.a == oracle_before.a && state.c == oracle_before.c);
  row.oracle_checked = true;
 });
 row.trial_valid = row.oracle_checked && row.common_a == 1 && row.full_ab == 1 && row.state_match == 1 && row.accepted;
 row.status = row.trial_valid ? "ok" : "validation_failed";
 row.phase = "complete";
 trace.emit(row, update.u, update.v, a, *delivered, zs, state.a, state.c);
}

int main(int argc, char **argv) {
 const auto start = Clock::now();
 if (argc == 2 && std::string(argv[1]) == "--help") {
  std::cout << "q61_bench --n N --r R [--t 2] [--seeds 101,211,307,401,503] [--reps 3] [--rounds 4] [--max-seconds 900] [--out PREFIX] [--trace-out PILOT.json]\n";
  return 0;
 }
 try {
  const Config c = parse(argc, argv);
  const Limit limit{start, c.max_seconds};
  Output output(c); Trace trace(c);
  std::string status = "completed", reason;
  int exit_code = 0;
  try {
   for (std::size_t si = 0; si < c.seeds.size(); ++si) {
    for (int rep = 0; rep < c.repeats; ++rep) {
     limit.check();
     Word setup_ns = 0;
     Matrix a0, b, c0;
     std::vector<Update> updates;
     std::array<State, 3> states;
     try { timed(setup_ns, [&] {
      std::mt19937_64 rng(c.seeds[si]);
      a0 = random_matrix(c.n, c.n, rng, limit);
      if (std::all_of(a0.a.begin(), a0.a.end(), [](Word x) { return x == 0; })) a0.a[0] = 1;
      b = unit_lu(c.n, rng, limit); c0 = mm(a0, b, limit);
      for (int j = 0; j < c.rounds; ++j)
       updates.push_back({random_matrix(c.n, c.r, rng, limit), random_matrix(c.n, c.r, rng, limit)});
      for (auto &state : states) { state.a = a0; state.c = c0; }
     }); } catch (const Deadline &) { output.setup(c.seeds[si], rep, setup_ns, "deadline"); throw; }
      catch (...) { output.setup(c.seeds[si], rep, setup_ns, "error"); throw; }
     output.setup(c.seeds[si], rep, setup_ns); trace.setup(c.seeds[si], rep, a0, b, c0);
     for (int round = 0; round < c.rounds; ++round) {
      for (int pos = 0; pos < 3; ++pos) {
       Row row;
       row.seed = c.seeds[si]; row.repeat = rep; row.round = round; row.setup_ns = setup_ns;
       row.method = int((si + std::size_t(rep + round + pos)) % 3);
       try { run_round(states[row.method], b, updates[round], c, limit, row, trace); }
       catch (const Deadline &) { row.status = "deadline"; output.emit(row); throw; }
       catch (const std::exception &) { row.status = "error"; output.emit(row); throw; }
       output.emit(row);
       if (!row.trial_valid) { status = "validation_failed"; throw std::runtime_error("round failed validation; retained raw row, no retry"); }
       limit.check();
      }
     }
    }
   }
  } catch (const Deadline &e) { status = "deadline"; reason = e.what(); exit_code = 124; }
  catch (const std::exception &e) { if (status == "completed") status = "error"; reason = e.what(); exit_code = status == "validation_failed" ? 1 : 2; }
  trace.finish(status); output.finish(status, reason, elapsed(start));
  if (exit_code) std::cerr << status << ": " << reason << '\n';
  return exit_code;
 } catch (const std::exception &e) {
  std::cerr << "configuration/output error: " << e.what() << '\n';
  return 2;
 }
}
