"""Scrape new UFC fights from ufcstats.com and append them to the gold dataset.

Finds completed events newer than the latest Event_Date in
ufc_gold_dataset_final.csv, parses each fight-details page into the gold
schema (full stats: KD, sig strikes, TD, control time, per-target and
per-position splits), and adds any unseen fighters to ufc_fighters_final.csv.
Elo and all career features are recomputed from the gold dataset by train.py,
so after this script finishes just run `python train.py`.

Usage: python update_data.py [--since YYYY-MM-DD] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).resolve().parent
GOLD_CSV = DATA_DIR / 'ufc_gold_dataset_final.csv'
FIGHTERS_CSV = DATA_DIR / 'ufc_fighters_final.csv'

BASE = 'http://www.ufcstats.com'
EVENTS_URL = BASE + '/statistics/events/completed?page=all'
REQUEST_DELAY = 0.8  # be polite

GOLD_COLUMNS = [
    'Fight_URL', 'Fighter_1', 'Fighter_2', 'Winner', 'Weight_Class', 'Method',
    'End_Round', 'End_Time', 'Total_Fight_Time_Sec', 'Time_Format',
    'F1_KD', 'F2_KD', 'F1_Sig_Landed', 'F1_Sig_Att', 'F2_Sig_Landed',
    'F2_Sig_Att', 'F1_TD_Landed', 'F2_TD_Landed', 'F1_TD_Att', 'F2_TD_Att',
    'F1_Sub_Att', 'F2_Sub_Att', 'F1_Ctrl_Sec', 'F2_Ctrl_Sec',
    'F1_Head', 'F2_Head', 'F1_Body', 'F2_Body', 'F1_Leg', 'F2_Leg',
    'F1_Distance', 'F2_Distance', 'F1_Clinch', 'F2_Clinch',
    'F1_Ground', 'F2_Ground', 'Event_Date',
]

FIGHTER_COLUMNS = [
    'Fighter_Name', 'Height', 'Weight', 'Reach', 'Stance', 'DOB',
    'Wins', 'Losses', 'Draws', 'SLpM', 'Str_Acc', 'SApM', 'Str_Def',
    'TD_Avg', 'TD_Acc', 'TD_Def', 'Sub_Avg', 'Fighter_URL',
]


class UFCStatsSession:
    """requests.Session that transparently solves ufcstats.com's
    sha256 proof-of-work browser check (POST nonce+n to /__c)."""

    CHALLENGE_RE = re.compile(r'var nonce="([^"]+)",\s*target=new Array\((\d+)\+1\)')

    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
        })
        self._last_request = 0.0

    def get(self, url: str, retries: int = 3) -> BeautifulSoup:
        # cookie from the PoW is host-specific; the dataset stores
        # no-www URLs, so normalize before fetching
        url = url.replace('http://ufcstats.com', BASE)
        for attempt in range(retries):
            wait = REQUEST_DELAY - (time.time() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            try:
                r = self.s.get(url, timeout=30)
                self._last_request = time.time()
                m = self.CHALLENGE_RE.search(r.text)
                if m:
                    self._solve_challenge(m.group(1), int(m.group(2)))
                    r = self.s.get(url, timeout=30)
                    self._last_request = time.time()
                r.raise_for_status()
                return BeautifulSoup(r.content, 'html.parser')
            except requests.RequestException as e:
                if attempt == retries - 1:
                    raise
                print(f'  retry {attempt + 1} for {url}: {e}')
                time.sleep(2 * (attempt + 1))

    def _solve_challenge(self, nonce: str, zeros: int) -> None:
        target = '0' * zeros
        n = 0
        while not hashlib.sha256(f'{nonce}:{n}'.encode()).hexdigest().startswith(target):
            n += 1
        self.s.post(BASE + '/__c', data={'nonce': nonce, 'n': n}, timeout=30)


def fight_id(url: str) -> str:
    return url.rstrip('/').split('/')[-1]


def _num(text: str) -> int:
    text = text.strip()
    return int(text) if text.isdigit() else 0


def _of_pair(text: str) -> tuple[int, int]:
    """'26 of 45' -> (26, 45); '---' -> (0, 0)."""
    m = re.match(r'(\d+)\s+of\s+(\d+)', text.strip())
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _mmss_to_sec(text: str) -> int:
    m = re.match(r'(\d+):(\d+)', text.strip())
    return int(m.group(1)) * 60 + int(m.group(2)) if m else 0


def _total_fight_sec(time_format: str, end_round: int, end_time_sec: int) -> int:
    """Sum the full rounds before the final one (lengths from the
    '(5-5-5)' part of the format) plus time into the final round."""
    m = re.search(r'\(([\d-]+)\)', time_format)
    if not m:
        return end_time_sec  # 'No Time Limit'
    round_mins = [int(x) for x in m.group(1).split('-')]
    full = sum(round_mins[:end_round - 1]) * 60 if end_round > 1 else 0
    return full + end_time_sec


def list_completed_events(session: UFCStatsSession) -> list[dict]:
    """All completed events, oldest first: {name, date, url}."""
    soup = session.get(EVENTS_URL)
    events = []
    for tr in soup.select('tr.b-statistics__table-row'):
        a = tr.find('a', class_='b-link')
        d = tr.find('span', class_='b-statistics__date')
        if not a or not d:
            continue
        events.append({
            'name': a.get_text(strip=True),
            'date': pd.to_datetime(d.get_text(strip=True)),
            'url': a['href'],
        })
    events.reverse()
    return events


def event_fight_urls(session: UFCStatsSession, event_url: str) -> list[str]:
    """Fight-details URLs for an event, prelims first (the gold dataset
    stores fights in card order: early prelims up to the main event)."""
    soup = session.get(event_url)
    urls = [tr['data-link']
            for tr in soup.select('tr.b-fight-details__table-row[data-link]')]
    urls.reverse()
    return urls


def parse_fight(session: UFCStatsSession, url: str, event_date: pd.Timestamp) -> dict | None:
    """Parse one fight-details page into a gold-schema row.
    Returns None for fights without a result yet."""
    soup = session.get(url)

    persons = soup.select('div.b-fight-details__person')
    if len(persons) != 2:
        return None
    names, statuses = [], []
    for p in persons:
        nm = p.find('a', class_='b-fight-details__person-link')
        st = p.find('i', class_='b-fight-details__person-status')
        names.append(nm.get_text(strip=True) if nm else '')
        statuses.append(st.get_text(strip=True) if st else '')
    if 'W' in statuses:
        winner = names[statuses.index('W')]
    elif statuses[0] in ('D', 'NC'):
        winner = 'Draw/NC'
    else:
        return None  # upcoming fight

    row = dict.fromkeys(GOLD_COLUMNS, 0)
    row['Fight_URL'] = 'http://ufcstats.com/fight-details/' + fight_id(url)
    row['Fighter_1'], row['Fighter_2'] = names
    row['Winner'] = winner
    row['Event_Date'] = str(event_date.date())

    title = soup.find('i', class_='b-fight-details__fight-title')
    row['Weight_Class'] = title.get_text(strip=True) if title else ''

    items = {}
    for it in soup.select('i.b-fight-details__text-item_first, i.b-fight-details__text-item'):
        txt = ' '.join(it.get_text().split())
        if ':' in txt:
            k, v = txt.split(':', 1)
            items[k.strip()] = v.strip()
    row['Method'] = items.get('Method', '')
    row['End_Round'] = _num(items.get('Round', '0'))
    row['End_Time'] = items.get('Time', '0:00')
    row['Time_Format'] = items.get('Time format', '')
    row['Total_Fight_Time_Sec'] = _total_fight_sec(
        row['Time_Format'], row['End_Round'], _mmss_to_sec(row['End_Time']))

    # stats tables: totals (KD/Sig/Td/Sub/Ctrl) and sig-strike breakdown
    # (Head/Body/Leg/Distance/Clinch/Ground); skip the per-round variants
    totals = breakdown = None
    for t in soup.find_all('table'):
        ths = [' '.join(th.get_text().split()) for th in t.find_all('th')]
        if 'Round 1' in ths:
            continue
        if 'Ctrl' in ths and totals is None:
            totals = t
        elif 'Head' in ths and breakdown is None:
            breakdown = t

    def cells(table):
        tr = table.find('tbody').find('tr')
        return [[' '.join(p.get_text().split()) for p in td.find_all('p')]
                for td in tr.find_all('td')]

    if totals is not None:
        c = cells(totals)
        # td order: Fighter, KD, Sig str, Sig %, Total str, Td, Td %, Sub att, Rev, Ctrl
        if c[0][0] != names[0]:  # keep stats aligned with the person order
            c = [[cell[1], cell[0]] for cell in c]
        for side, f in ((0, 'F1'), (1, 'F2')):
            row[f'{f}_KD'] = _num(c[1][side])
            row[f'{f}_Sig_Landed'], row[f'{f}_Sig_Att'] = _of_pair(c[2][side])
            row[f'{f}_TD_Landed'], row[f'{f}_TD_Att'] = _of_pair(c[5][side])
            row[f'{f}_Sub_Att'] = _num(c[7][side])
            row[f'{f}_Ctrl_Sec'] = _mmss_to_sec(c[9][side])

    if breakdown is not None:
        c = cells(breakdown)
        # td order: Fighter, Sig str, Sig %, Head, Body, Leg, Distance, Clinch, Ground
        if c[0][0] != names[0]:
            c = [[cell[1], cell[0]] for cell in c]
        for ci, col in ((3, 'Head'), (4, 'Body'), (5, 'Leg'),
                        (6, 'Distance'), (7, 'Clinch'), (8, 'Ground')):
            row[f'F1_{col}'] = _of_pair(c[ci][0])[0]
            row[f'F2_{col}'] = _of_pair(c[ci][1])[0]

    # fighter-details links, used to scrape fighters new to the roster
    row['_fighter_urls'] = [
        p.find('a', class_='b-fight-details__person-link')['href']
        if p.find('a', class_='b-fight-details__person-link') else None
        for p in persons
    ]
    return row


def parse_fighter(session: UFCStatsSession, url: str) -> dict:
    """Parse a fighter-details page into the fighters-CSV schema."""
    soup = session.get(url)
    row = dict.fromkeys(FIGHTER_COLUMNS, '')
    row['Fighter_Name'] = soup.find(
        'span', class_='b-content__title-highlight').get_text(strip=True)
    row['Fighter_URL'] = 'http://ufcstats.com/fighter-details/' + fight_id(url)

    rec = soup.find('span', class_='b-content__title-record').get_text(strip=True)
    m = re.search(r'(\d+)-(\d+)-(\d+)', rec)
    if m:
        row['Wins'], row['Losses'], row['Draws'] = (int(g) for g in m.groups())

    label_to_col = {
        'Height': 'Height', 'Weight': 'Weight', 'Reach': 'Reach',
        'STANCE': 'Stance', 'DOB': 'DOB', 'SLpM': 'SLpM',
        'Str. Acc.': 'Str_Acc', 'SApM': 'SApM', 'Str. Def': 'Str_Def',
        'TD Avg.': 'TD_Avg', 'TD Acc.': 'TD_Acc', 'TD Def.': 'TD_Def',
        'Sub. Avg.': 'Sub_Avg',
    }
    for li in soup.select('li.b-list__box-list-item'):
        txt = ' '.join(li.get_text().split())
        if ':' not in txt:
            continue
        k, v = txt.split(':', 1)
        k, v = k.strip(), v.strip()
        if k in label_to_col and v != '--':
            row[label_to_col[k]] = v
    if row['DOB']:
        row['DOB'] = str(pd.to_datetime(row['DOB']).date())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', help='scrape events on/after this date '
                                    '(default: latest Event_Date in the gold dataset)')
    ap.add_argument('--dry-run', action='store_true',
                    help='scrape and report, but do not write the CSVs')
    args = ap.parse_args()

    gold = pd.read_csv(GOLD_CSV)
    fighters = pd.read_csv(FIGHTERS_CSV)
    known_fights = {fight_id(u) for u in gold['Fight_URL']}
    known_fighter_ids = {fight_id(u) for u in fighters['Fighter_URL'].dropna()}
    known_fighter_names = set(fighters['Fighter_Name'])

    since = pd.to_datetime(args.since) if args.since else pd.to_datetime(gold['Event_Date']).max()
    today = pd.Timestamp.today().normalize()
    print(f'gold dataset: {len(gold)} fights through {since.date()}')

    session = UFCStatsSession()
    events = [e for e in list_completed_events(session)
              if since <= e['date'] <= today]
    print(f'{len(events)} completed events on/after {since.date()}')

    new_rows, new_fighter_urls = [], []
    for ev in events:
        urls = event_fight_urls(session, ev['url'])
        fresh = [u for u in urls if fight_id(u) not in known_fights]
        print(f"{ev['date'].date()}  {ev['name']}: "
              f"{len(fresh)} new of {len(urls)} fights")
        for u in fresh:
            row = parse_fight(session, u, ev['date'])
            if row is None:
                print(f'  no result yet, skipping {u}')
                continue
            for name, furl in zip((row['Fighter_1'], row['Fighter_2']),
                                  row.pop('_fighter_urls')):
                if furl and name not in known_fighter_names \
                        and fight_id(furl) not in known_fighter_ids:
                    new_fighter_urls.append(furl)
                    known_fighter_ids.add(fight_id(furl))
            known_fights.add(fight_id(u))
            new_rows.append(row)

    new_fighters = []
    for furl in new_fighter_urls:
        f = parse_fighter(session, furl)
        print(f"  new fighter: {f['Fighter_Name']}")
        new_fighters.append(f)

    print(f'\nscraped {len(new_rows)} new fights, {len(new_fighters)} new fighters')
    if args.dry_run or not new_rows:
        return

    gold = pd.concat([gold, pd.DataFrame(new_rows)[GOLD_COLUMNS]], ignore_index=True)
    gold.to_csv(GOLD_CSV, index=False)
    if new_fighters:
        fighters = pd.concat(
            [fighters, pd.DataFrame(new_fighters)[FIGHTER_COLUMNS]], ignore_index=True)
        fighters.to_csv(FIGHTERS_CSV, index=False)
    print(f'gold dataset now {len(gold)} fights through '
          f"{pd.to_datetime(gold['Event_Date']).max().date()}; "
          f'fighters table now {len(fighters)} rows')
    print('next: python train.py to recompute Elo, features and the model')


if __name__ == '__main__':
    main()
