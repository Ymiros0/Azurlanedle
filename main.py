from math import log2
import statistics
import unicodedata
import json
from prompt_toolkit import prompt
from prompt_toolkit.completion import Completer, Completion
import textwrap
import argparse
from time import time
from random import Random

try:
    bit_count = int.bit_count
except AttributeError:
    # Fallback for older Python
    def bit_count(x: int) -> int:
        count = 0
        while x:
            x &= x - 1
            count += 1
        return count


with open("dle_data.json", encoding = "utf-8") as f:
	data = json.load(f)

with open("events.json", encoding = "utf-8") as f:
	events = json.load(f)

with open("cruise.json", encoding = "utf-8") as f:
	cruise = json.load(f)

with open("showdown.json", encoding = "utf-8") as f:
	meta = json.load(f)

with open("research.json", encoding = "utf-8") as f:
	research = json.load(f)

try:
	with open("history.json", encoding = "utf-8") as f:
		hist = json.load(f)
except FileNotFoundError:
	hist = {}

with open("true_history.json", encoding="utf-8") as f:
	true_history = json.load(f)


FIELDS_ORDER = ["name", "rarity", "nation", "hull", "class", "timer", "event", "VA"]
ids = list(data.keys())[:3000]            # original ids (strings like "20204")
n = len(ids)
id_to_idx = {id_: i for i, id_ in enumerate(ids)}
SHIPS = [data[id_] for id_ in ids]  # indexed list of ship dicts
name_to_idx = {SHIPS[i]["name"]: i for i in range(len(SHIPS))}
FULL_MASK = (1 << n) - 1

# ======================
# Helpers: bitmask <-> ids, sizes
# ======================
def mask_to_ids(mask):
	"""Return list of indices set in mask (in ascending order)."""
	out = []
	m = mask
	while m:
		lsb = m & -m
		idx = lsb.bit_length() - 1
		out.append(idx)
		m ^= lsb
	return out

def single_index_from_mask(mask):
	"""Return index of single set bit; undefined if not exactly one bit."""
	return (mask & -mask).bit_length() - 1

def normalise_compare(compare_dict):
	return tuple(compare_dict.get(k) for k in FIELDS_ORDER)
		
# Top-level bot process function for Windows compatibility
def bot_process_worker(queue=None):
	# Precompute feedback groups: for each guess_id, map feedback_tuple -> bitmask of solution ids
	guess_fb_map = [None] * n  # guess_fb_map[gid] = { fb_tuple: bitmask_of_sids }
	for gid in range(n):
		mapping = {}
		gdict = SHIPS[gid]
		for sid in range(n):
			sdict = SHIPS[sid]
			fb_dict = compare_ship(sdict, gdict)  # note: solution first, guess second
			if fb_dict == {'name': 'Yes', 'nation': 'Yes', 'rarity': 'Yes', 'hull': 'Yes', 'class': 'Yes', 'VA': 'Yes', 'timer': 'Yes', 'event': 'Yes'}:
				continue
			fb = normalise_compare(fb_dict)
			mapping.setdefault(fb, 0)
			mapping[fb] |= (1 << sid)
		guess_fb_map[gid] = mapping
	if queue is not None:
		queue.put(guess_fb_map)
	return guess_fb_map

def eval_skill_entropy(mask, guess_fb_map):
	cnt = bit_count(mask)
	if cnt < 2:
		if cnt == 0:
			return {}
		if cnt == 1:
			idx = single_index_from_mask(mask)
			return {idx: 100}
	guess_ids = mask_to_ids(FULL_MASK)
	scores = {}
	best_score = -1
	worst_score = 1000 #Technically not future proof, but it'll do for the next quadrillion years with the rate new ships are added
	for gid in guess_ids:
		H = 0
		for fb_mask in guess_fb_map[gid].values():
			child_mask = mask & fb_mask
			if not child_mask:
				continue
			p = bit_count(child_mask)/cnt
			H -= p* log2(p)
		if mask & (1 << gid):
			H -= 1/cnt*log2(1/cnt)
		else:
			H /= 2
		if H > best_score:
			best_score = H
		if H < worst_score:
			worst_score = H
		scores[gid] = H
	if best_score > 0:
		for k,v in scores.items():
			scores[k] = round(100*(v-worst_score)/(best_score-worst_score))
	else:
		for k in scores:
			scores[k] = 0
	return scores

def eval_luck_entropy(mask, guess_fb_map, solution):
	cnt = bit_count(mask)
	if cnt < 2:
		if cnt == 0:
			return {}
		if cnt == 1:
			idx = single_index_from_mask(mask)
			return {idx: 50}
	guess_ids = mask_to_ids(FULL_MASK)
	scores = {}
	sol_mask = 1 << solution
	for gid in guess_ids:
		if gid == solution:
			scores[gid] = 100
			continue
		l = cnt
		u = 0
		if 1 << gid & mask:
			l = 0
		for fb_mask in guess_fb_map[gid].values():
			child_mask = mask & fb_mask
			s = bit_count(child_mask)
			if s == 0 and gid != solution:
				continue
			if sol_mask & fb_mask:
				x = s
			if s < l:
				l = s
			if s > u:
				u = s
		if u > l:
			scores[gid] = int(100*(u-x)/(u-l))
		elif u == 0:
			scores[gid] = 100
		else:
			scores[gid] = 50
	return scores

def run_bot_eval(guess_fb_map, solution, guesses):
	mask = filtered_mask
	guesses = [name_to_idx[i] for i in guesses]
	solution_id = name_to_idx[solution]
	"""Evaluate a sequence of guesses with entropy-based Skill/Luck and info gains.
	Returns a list of dicts (one per guess) and the final mask.
	Each dict: {
		'guess': gid,
		'skill': 0..100,
		'luck': 0..100,
		'remaining': candidates_after_guess
	}
	"""
	out = []
	cur_mask = mask
	for gid in guesses:
		# step down to the actual branch
		next_mask = 0
		for fb_mask in guess_fb_map[gid].values(): #TODO: Just use the actual compare value? You know the solution
			child_mask = cur_mask & fb_mask
			if (child_mask >> solution_id) & 1:
				next_mask = child_mask
				break
		scores = eval_skill_entropy(cur_mask, guess_fb_map)
		skl = scores.get(gid, 0)
		lck = eval_luck_entropy(cur_mask, guess_fb_map, solution_id).get(gid, 50)
		bot_guess = sorted(scores, key=scores.get, reverse=True)[0]
		remain = bit_count(next_mask)
		out.append({
			'guess': SHIPS[gid]["name"],
			'skill': skl,
			'luck': lck,
			'botg': SHIPS[bot_guess]["name"],
			'remaining': ", ".join(SHIPS[i]["name"] for i in mask_to_ids(next_mask)) if remain < 10 else remain
		})
		cur_mask = next_mask if next_mask else cur_mask
	return out, cur_mask

def get_max_skill_guess(mask):
	scores = eval_skill_entropy(mask, guess_fb_map)
	ordered = sorted(scores, key=scores.get, reverse=True)
	#print([(SHIPS[i]["name"], scores[i]) for i in ordered[:10]])
	return ordered[0]

def sim_play(solution,mask = FULL_MASK):
	solution_id = name_to_idx[solution]
	guesses = []
	while mask:
		guess = get_max_skill_guess(mask)
		if guess == solution_id:
			guesses.append([SHIPS[guess], {'name': 'Yes', 'nation': 'Yes', 'rarity': 'Yes', 'hull': 'Yes', 'class': 'Yes', 'VA': 'Yes', 'timer': 'Yes', 'event': 'Yes'}])
			break
		response = normalise_compare(compare_ship(get_data(solution), SHIPS[guess]))
		fb_mask = guess_fb_map[guess][response]
		mask &= fb_mask
		guesses.append([SHIPS[guess], dict(zip(FIELDS_ORDER, response))])
	print_guess_table(guesses)


	

def normalize_for_compare(s: str) -> str:
	# Normalize to NFKD (compatibility decomposition)
	# so "Ä" becomes "A" + "¨" and "µ" becomes "μ"
	s = unicodedata.normalize("NFKD", s)
	# Remove combining marks (accents, umlauts, etc.)
	s = "".join(c for c in s if not unicodedata.combining(c))
	# Apply case folding for Unicode-safe lowercasing
	s = s.replace("'", "").replace(".", "").replace("(", "").replace(')', '')
	return s.casefold()

def matches(a: str, b: str) -> bool:
	return normalize_for_compare(a) == normalize_for_compare(b)

def get_date(timer, event):
	if event == "No Event":
		return
	table = {
		"Research": research,
		"META Showdown": meta,
		"Cruise Missions": cruise
	}
	lookup = table.get(timer, events)
	return lookup[event]

def parse_timer(timer):
	if ':' in timer:
		return int(timer.replace(':',''))
	if timer == "Drop Only":
		return "Cannot be constructed"
	return timer

def compare_ship(solution, guess):
	result = {}

	# Simple Yes/No fields
	simple_fields = ["name", "nation", "rarity", "hull", "class", "VA"]
	for field in simple_fields:
		result[field] = "Yes" if solution[field] == guess[field] else "No"

	# Timer comparison
	sol_timer = parse_timer(solution["timer"])
	guess_timer = parse_timer(guess["timer"])
	if sol_timer == guess_timer:
		result["timer"] = "Yes"
	elif isinstance(sol_timer, str) or isinstance(guess_timer, str):
		result["timer"] = "No"
	elif sol_timer > guess_timer:
		result["timer"] = "↑"
	else:
		result["timer"] = "↓"

	# Event comparison
	sol_date = get_date(sol_timer, solution["event"])
	guess_date = get_date(guess_timer, guess["event"])
	if sol_date == guess_date:
		result["event"] = "Yes"
	elif sol_date is None or guess_date is None:
		result["event"] = "No"
	elif sol_date > guess_date:
		result["event"] = "↑"
	else:
		result["event"] = "↓"

	return result

def get_data(ship):
	if isinstance(ship, str):
		for i in names:
			if matches(i, ship):
				return data[names[i]]
	if isinstance(ship, int):
		return data.get(str(ship))

def color_bg(text, correct):
	if correct == "Yes":
		return f"\033[30;42m{text}\033[0m"  # Green bg
	if correct:
		return f"\033[97;41m{text}\033[0m"  # Red bg
	return text

def print_guess_eval(results):
	"""
	Pretty-print a table of guesses with skill/luck/bot guess/remaining ships.
	Expects a list of dicts like:
	{'guess': ..., 'skill': ..., 'luck': ..., 'botg': ..., 'remaining': ...}
	"""
	# Header
	header = {"guess": "Guess", "skill": "Skill", "luck": "Luck", "botg": "Bot guess", "remaining": "Remaining ships"}
	fields = ['guess', "skill", "luck", "botg", "remaining"]
	widths = [max(len(str(r[k])) for r in results) for k in fields]
	widths = [max(w, len(h)) for w, h in zip(widths, header.values())]

	def fmt_row(row):
		return " | ".join(str(val).center(widths[i]) for i, val in enumerate(row))

	# Print
	print(fmt_row(header.values()))
	print("-+-".join("-" * w for w in widths))
	for r in results:
		row = [
			r['guess'],
			r['skill'],
			r['luck'],
			r['botg'],
			r['remaining']
		]
		print(fmt_row(row))

def print_guess_table(guesses):
	# Column headers and widths
	headers = {"name": "Guess", "rarity": "Rarity", "hull": "Hull", "nation": "Nation", "class": "Class", "timer": "Timer", "event": "Event", "VA": "Voice Actor"}
	headerl = list(headers.keys())
	col_widths = {
		"name": 20,
		"rarity": 12,
		"hull": 4,
		"nation": 15,
		"class": 15,
		"timer": 11,
		"event": 25,
		"VA": 20
	}

	def wrap_and_center(text, width):
		"""Wrap text to a given width and center each line."""
		wrapped = textwrap.wrap(str(text), width=width) or [""]
		return [line.center(width) for line in wrapped]

	def format_row(row_data, result={}):
		"""Format a row where each cell may have multiple wrapped lines."""
		wrapped_columns = []
		for h in headers:
			r = result.get(h)
			ad = f" {r}" if r in ("↓", "↑") else ""
			t = row_data.get(h, "").replace("Drop Only", "Cannot be constructed") + ad
			t = [color_bg(i, r) for i in wrap_and_center(t, col_widths[h])]
			wrapped_columns.append(t)
		max_lines = max(len(col) for col in wrapped_columns)


		lines = []
		for i in range(max_lines):
			line_parts = [
				wrapped_columns[j][i] if i < len(wrapped_columns[j]) else " " * col_widths[headerl[j]]
				for j in range(len(headers))
			]
			lines.append(" | ".join(line_parts))
		return "\n".join(lines)

	# Print header
	print(format_row(headers))
	print("-" * (sum(col_widths.values()) + 3 * (len(headers) - 1)))

	# Print all guesses
	for guess, res in guesses:
		print(format_row(guess, res))

names = {}
for k,v in data.items():
	names[v["name"]] = k

class NormalizingCompleter(Completer):
    """
    A prompt_toolkit Completer that matches normalized forms but shows original candidates.
    - candidates: iterable of original strings
    - show_all_on_empty: if True, pressing Tab with empty input will list all candidates
    """
    def __init__(self, candidates, show_all_on_empty=False):
        self.candidates = list(dict.fromkeys(candidates))  # remove exact duplicates while preserving order
        # precompute normalized forms
        self._cached = [(cand, normalize_for_compare(cand)) for cand in self.candidates]
        self.show_all_on_empty = show_all_on_empty

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        ntext = normalize_for_compare(text)

        # If user typed nothing and we shouldn't show everything, exit.
        if ntext == "" and not self.show_all_on_empty:
            return

        # Score and collect matches:
        # score 0 = candidate normalized startswith input (best)
        # score 1 = candidate normalized contains input
        matches = []
        for orig, norm in self._cached:
            if ntext == "":
                # if empty and we allow showing all, treat as contains match
                matches.append((1, orig))
            elif norm.startswith(ntext):
                matches.append((0, orig))
            elif ntext in norm:
                matches.append((1, orig))

        # sort matches: best matches first, then alphabetically (you can tweak)
        matches.sort(key=lambda s: (s[0], s[1].casefold()))

        # Replace entire buffer with the chosen completion
        start_pos = -len(text) if len(text) > 0 else 0

        for _, orig in matches:
            yield Completion(orig, start_position=start_pos)

def get_guess():
	completer = NormalizingCompleter(filtered_names, show_all_on_empty=False)
	user_input = prompt("Guess: ", completer=completer)
	return user_input


def get_ship_of_the_day(day, pool):
	rng = Random(day).choice(sorted(pool))
	return rng


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Play the Azur Lane DLE game.")
	parser.add_argument("-e", "--easy", action="store_true", help="for those who can't think (Only enable possible solutions as guesses)")
	parser.add_argument("-n", "--no-repeats", action="store_true", help="check history for no reapeats in suggestions and bot play")
	parser.add_argument("-c", "--compact", action="store_true", help="enable compact output mode (Good when colours don't work)")
	parser.add_argument("-b", "--no-bot", action="store_true", help="disable bot mode (don't show bot's play and evaluation)")
	parser.add_argument("-B", "--no-repeats-bot-only", action="store_true", help="check history for no repeats in bot play only")
	args = parser.parse_args()
	easy = args.easy
	no_repeats = args.no_repeats
	coloured = not args.compact
	bot_mode = not args.no_bot
	no_repeats_bot_only = args.no_repeats_bot_only #TODO: implement this

	d = 24*3600
	today = int((time()-7*3600)//d)
	history_ships = set(v for k,v in hist.items() if k != str(today))
	filtered_names = {name:idx for name, idx in names.items() if name not in history_ships or not no_repeats}
	filtered_mask = FULL_MASK
	if no_repeats:
		for name in filtered_names:
			idx = name_to_idx[name]
			filtered_mask ^= (1 << idx)


	print(f"Booting Azur Lane DLE... {len(filtered_names)} ships available for guessing.")

	# Calculate true_history until today
	discrepancies = []
	pool = set(names)-set(true_history.values())
	for day in range(20309, today+1):
		day_str = str(day)
		true_ship = true_history.get(day_str)
		user_ship = hist.get(day_str)
		if true_ship is not None:
			if user_ship is not None and true_ship != user_ship:
				discrepancies.append((day_str, user_ship, true_ship))
		else:
			true_ship = get_ship_of_the_day(day, pool)
			pool.remove(true_ship)
			if user_ship is not None and true_ship != user_ship:
				discrepancies.append((day_str, user_ship, true_ship))
				hist.pop(day_str)

	if discrepancies:
		print(f"{len(discrepancies)} discrepancies found between your history and the true history, updating history to reflect true history!")
		# for day_str, user_ship, true_ship in discrepancies:
		# 	print(f"Day {day_str}: Your history: {user_ship} | True: {true_ship}")

	# Update history with true_history until today
	for day in range(20309, today+1):
		day_str = str(day)
		if day_str in true_history:
			hist[day_str] = true_history[day_str]

	last_played = max(int(k) for k in hist) if hist else 20308

	if today == last_played:
		print("You already played today.")
		h = 24-(time()-7*3600)%d/3600
		print(f"Next game available in {int(h):0>2}h{int(h%1*60):0>2}m{int((h*3600)%60):0>2}s")
		solution = hist[str(today)]
	else:
		used = set(hist.values())
		pool = set(names)-used
		for i in range(last_played+1, today+1):
			ship = get_ship_of_the_day(i, pool)
			hist[str(i)] = ship
			pool.remove(ship)
		solution = ship

	solution_obj = get_data(solution)
	correct = False
	guesses = 0
	history_list = []
	player_guess_names = []

	# bot_result_queue = multiprocessing.Queue()
	# bot_proc = multiprocessing.Process(target=bot_process_worker, args=(bot_result_queue,))
	# bot_proc.start()
	# Compute feedback map in main thread

	# try:
	while not correct:
		try: guess = get_guess()
		except KeyboardInterrupt:
			print("Ctrl+C is disabled. Type 'Abort' to exit.")
			continue
		if guess in ("Abort", "abort"):
			print("Ending Game.")
			# bot_proc.terminate()
			exit()
		gd = get_data(guess)
		if not gd:
			print("Ship not found:", guess)
			continue
		res = compare_ship(solution_obj, gd)
		if coloured:
			history_list.append([gd, res])
			print_guess_table(history_list)
		else:
			print(f'Rarity: {res["rarity"]} | Hull Type: {res["hull"]} | Nation: {res["nation"]} | Class: {res["class"]} | Timer: {res["timer"]} | Event: {res["event"]} | VA: {res["VA"]}')
		guesses += 1
		player_guess_names.append(gd["name"])
		filtered_names.pop(gd["name"])
		if easy:
			filtered_names = {name:idx for name, idx in filtered_names.items() if compare_ship(get_data(name), gd) == res}
		correct = res["name"] == "Yes"

	print("YOU WIN!")
	print(f"It took you {guesses} guesses.")
	if easy:
		print("If you can even count easy mode as a win...")

	with open("history.json", "w", encoding="utf-8") as f:
		json.dump(hist, f, indent=2)

	# Prompt for evaluation
	eval_choice = bot_mode or input("Would you like to see an evaluation of your guesses? (Y/N): ").strip().lower() in ("y", "yes")
	if eval_choice:
		filtered_mask = FULL_MASK
		if no_repeats or no_repeats_bot_only:
			filtered_names = {name:idx for name, idx in names.items() if name in history_ships}# and name not in player_guess_names}
			for name in filtered_names:
				idx = name_to_idx[name]
				filtered_mask ^= (1 << idx)
		guess_fb_map = bot_process_worker()
		# bot_proc.join()
		# guess_fb_map = bot_result_queue.get()
		# Get indices for player's guesses
		eval_results, mask = run_bot_eval(guess_fb_map, solution, player_guess_names)
		print_guess_eval(eval_results)
		if len(player_guess_names) == 1:
			print("Bloody hell, a perfect shot! You solved it in one guess... Are you perhaps cheating?!")
		elif all(i["skill"] == 100 for i in eval_results):
			print("You played like a true master, matching the optimal path!")
		elif statistics.mean(i["luck"] for i in eval_results) > 80:
			print("You won't get away with being this lucky next time, baka!")
		elif statistics.mean(i["skill"] for i in eval_results) < 20:
			print("Even Sandy would've solved this better than you. Did you even try?")
		elif statistics.mean(i["skill"] for i in eval_results) > 50:
			print("Hmph! At least you tried I guess. But I expect better next time.")
		else:
			print("L + ratio + skill issue + git gud! Try again tomorrow!")
		print("\nThis is how I would have played by the way:")
		sim_play(solution, filtered_mask)
	# except Exception as e:
	# 	print(f"An error occurred: {e}")
	# 	bot_proc.terminate()
	# 	exit(1)