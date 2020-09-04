"""
Implement clean_url function
"""
# pylint: disable=too-many-boolean-expressions
import re
from typing import Any, Tuple, Union

import dask.dataframe as dd
import dask
import numpy as np
import pandas as pd

from ..eda.utils import to_dask

NULL_VALUES = {
    np.nan,
    float("NaN"),
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "n/a",
    "nan",
    "null",
    "",
    "None"
}

user_regex = re.compile(
    # dot-atom
    r"(^[-!#$%&'*+/=?^_`{}|~0-9A-Z]+"
    r"(\.[-!#$%&'*+/=?^_`{}|~0-9A-Z]+)*$"
    # quoted-string
    r'|^"([\001-\010\013\014\016-\037!#-\[\]-\177]|'
    r"""\\[\001-\011\013\014\016-\177])*"$)""",
    re.IGNORECASE
)
domain_regex = re.compile(
    # domain
    r'(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+'
    r'(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?$)'
    # literal form, ipv4 address (SMTP 4.1.3)
    r'|^\[(25[0-5]|2[0-4]\d|[0-1]?\d?\d)'
    r'(\.(25[0-5]|2[0-4]\d|[0-1]?\d?\d)){3}\]$',
    re.IGNORECASE)
domain_whitelist = ['localhost']

domains = [
  #Default domains included#
  "aol.com", "att.net", "comcast.net", "facebook.com", "gmail.com", "gmx.com", "googlemail.com",
  "google.com", "hotmail.com", "hotmail.co.uk", "mac.com", "me.com", "mail.com", "msn.com",
  "live.com", "sbcglobal.net", "verizon.net", "yahoo.com", "yahoo.co.uk",

  #Other global domains
  "email.com", "fastmail.fm", "games.com", "gmx.net", "hush.com", "hushmail.com", "icloud.com",
  "iname.com", "inbox.com", "lavabit.com", "love.com", "outlook.com", "pobox.com", "protonmail.ch", "protonmail.com", "tutanota.de", "tutanota.com", "tutamail.com", "tuta.io",
 "keemail.me", "rocketmail.com", "safe-mail.net", "wow.com", "ygm.com",
  "ymail.com", "zoho.com", "yandex.com",

  #United States ISP domains
  "bellsouth.net", "charter.net", "cox.net", "earthlink.net", "juno.com",

  #British ISP domains
  "btinternet.com", "virginmedia.com", "blueyonder.co.uk", "freeserve.co.uk", "live.co.uk",
  "ntlworld.com", "o2.co.uk", "orange.net", "sky.com", "talktalk.co.uk", "tiscali.co.uk",
  "virgin.net", "wanadoo.co.uk", "bt.com",

  #Domains used in Asia
  "sina.com", "sina.cn", "qq.com", "naver.com", "hanmail.net", "daum.net", "nate.com", "yahoo.co.jp", "yahoo.co.kr", "yahoo.co.id", "yahoo.co.in", "yahoo.com.sg", "yahoo.com.ph", "163.com", "yeah.net", "126.com", "21cn.com", "aliyun.com", "foxmail.com",

  #French ISP domains
  "hotmail.fr", "live.fr", "laposte.net", "yahoo.fr", "wanadoo.fr", "orange.fr", "gmx.fr", "sfr.fr", "neuf.fr", "free.fr",

  #German ISP domains
  "gmx.de", "hotmail.de", "live.de", "online.de", "t-online.de", "web.de", "yahoo.de",

  #Italian ISP domains
  "libero.it", "virgilio.it", "hotmail.it", "aol.it", "tiscali.it", "alice.it", "live.it", "yahoo.it", "email.it", "tin.it", "poste.it", "teletu.it",

  #Russian ISP domains
  "mail.ru", "rambler.ru", "yandex.ru", "ya.ru", "list.ru",

  #Belgian ISP domains
  "hotmail.be", "live.be", "skynet.be", "voo.be", "tvcablenet.be", "telenet.be",

  #rgentinian ISP domains
  "hotmail.com.ar", "live.com.ar", "yahoo.com.ar", "fibertel.com.ar", "speedy.com.ar", "arnet.com.ar",

  #Domains used in Mexico
  "yahoo.com.mx", "live.com.mx", "hotmail.es", "hotmail.com.mx", "prodigy.net.mx",

  #Domains used in Canada
  "yahoo.ca", "hotmail.ca", "bell.net", "shaw.ca", "sympatico.ca", "rogers.com",

  #Domains used in Brazil
  "yahoo.com.br", "hotmail.com.br", "outlook.com.br", "uol.com.br", "bol.com.br", "terra.com.br", "ig.com.br", "itelefonica.com.br", "r7.com", "zipmail.com.br", "globo.com", "globomail.com", "oi.com.br"
]
        
nearbykeys = {
'a': ['q','w','s','x','z'],
'b': ['v','g','h','n'],
'c': ['x','d','f','v'],
'd': ['s','e','r','f','c','x'],
'e': ['w','s','d','r'],
'f': ['d','r','t','g','v','c'],
'g': ['f','t','y','h','b','v'],
'h': ['g','y','u','j','n','b'],
'i': ['u','j','k','o'],
'j': ['h','u','i','k','n','m'],
'k': ['j','i','o','l','m'],
'l': ['k','o','p'],
'm': ['n','j','k','l'],
'n': ['b','h','j','m'],
'o': ['i','k','l','p'],
'p': ['o','l'],
'q': ['w','a','s'],
'r': ['e','d','f','t'],
's': ['w','e','d','x','z','a'],
't': ['r','f','g','y'],
'u': ['y','h','j','i'],
'v': ['c','f','g','v','b'],
'w': ['q','a','s','e'],
'x': ['z','s','d','c'],
'y': ['t','g','h','u'],
'z': ['a','s','x'],
' ': ['c','v','b','n','m']
}

STATS = {"cleaned": 0, "null": 0, "unknown": 0}

def clean_email(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    split: bool = False,
    inplace: bool = False,
    pre_clean: bool = True,
    fix_domain: bool = False,
    error: str = 'coerce'
) -> pd.DataFrame:
    """
    This function cleans emails

    Parameters
    ----------
    df
        pandas or Dask DataFrame
    column
        column name
    split
        If True, split a column containing username and domain name
        into one column for username and one column for domain name
    inplace
        If True, delete the given column with dirty data, else, create a new
        column with cleaned data.
    pre_clean
        If True, apply basic text clean (like removing whitespaces) before 
        verifying and clean values.
    fix_domain
        If True, fix small typos in domain input
    error
        Specify ways to deal with broken value
        {'ignore', 'coerce'}, default 'coerce'
        'ignore': just return the initial input
        #'raise': raise an exception when there is broken value
        'coerce': set invalid value to NaN
    """
    # pylint: disable=too-many-arguments

    reset_stats()

    df = to_dask(df)
    # specify the metadata for dask apply
    meta = df.dtypes.to_dict()
    if split:
        meta.update(zip(("username", "domain"), (str, str)))
    else:
        meta[f"{column}_clean"] = str

    df = df.apply(
        format_email,
        args=(column, split, pre_clean, fix_domain, error),
        axis=1,
        meta=meta,
    )

    if inplace:
        df = df.drop(columns=[column])

    df, nrows = dask.compute(df, df.shape[0])

    report(nrows)

    return df

def format_email(
    row: pd.Series, col: str, split: bool, pre_clean: bool, fix_domain: bool, error: str,
) -> pd.Series:
    """
    Function to transform an email address into clean format
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements,too-many-return-statements
    if row[col] in NULL_VALUES:
        return not_email(row, col, split, "null",error)
    
    if '@' not in row[col]:
        return not_email(row, col, split, "unknown",error)

    user_part = ""
    domain_part = ""
    
    #pre-cleaning email text by removing all whitespaces

    if pre_clean == True:
        cleaned_address = re.sub(r'(\s|\u180B|\u200B|\u200C|\u200D|\u2060|\uFEFF)+', '', \
               str(row[col]))
    
        user_part, domain_part = cleaned_address.rsplit('@', 1)
    else:
        user_part, domain_part = str(row[col]).rsplit('@', 1)
    
    if not user_regex.match(user_part):
        return not_email(row, col, split, "unknown",error)
    
    if len(user_part.encode("utf-8")) > 64:
        return not_email(row, col, split, "unknown",error)
    
    if domain_part not in domain_whitelist and not domain_regex.match(domain_part):
        # Try for possible IDN domain-part
        try:
            initial_domain_part = domain_part
            domain_part = domain_part.encode('idna').decode('ascii')

            if domain_regex.match(domain_part) == None:
                return not_email(row, col, split, "unknown",error)
            else:
                if split:
                    row["username"], row["domain"] = str(user_part).lower(), str(initial_domain_part)
                else:
                    row[f"{col}_clean"] = str(user_part).lower() + "@" + str(initial_domain_part)
                return row
        except UnicodeError:
            return not_email(row, col, split, "unknown",error)

    #fix domain by detecting typos in advance
    if fix_domain == True:
        domains_map = set(domains)
        if domain_part.lower() not in domains_map:
            for i,c in enumerate(domain_part):
                if (domain_part[0:i] + domain_part[i+1:]).lower() in domains_map:
                    domain_part = (domain_part[0:i] + domain_part[i+1:]).lower()
                    break
                for new_c in "abcdefghijklmnopqrstuvwxyz":
                    if (domain_part[0:i+1] + new_c + domain_part[i+1:]).lower() in domains_map:
                        domain_part = (domain_part[0:i+1] + new_c + domain_part[i+1:]).lower()
                        break
                if i<len(domain_part)-1:
                    if (domain_part[0:i] + domain_part[i+1] + c + domain_part[i+2:]).lower() in domains_map:
                        domain_part = (domain_part[0:i] + domain_part[i+1] + c + domain_part[i+2:]).lower()
                        break
                if c in nearbykeys:
                    for c_p in nearbykeys[c]:
                        if (domain_part[0:i] + c_p + domain_part[i+1:]).lower() in domains_map:
                            domain_part = (domain_part[0:i] + c_p + domain_part[i+1:]).lower()
                            break
                
        
    if split:
        row["username"], row["domain"] = str(user_part).lower(), str(domain_part).lower()
    else:
        row[f"{col}_clean"] = str(user_part).lower() + "@" + str(domain_part).lower()

    return row

def validate_email(
    x: Union[str, float, pd.Series, Tuple[float, float]]
) -> Union[bool, pd.Series]:
    """
    This function validates emails

    Parameters
    ----------
    x
        pandas Series of emails or email instance
    """

    if isinstance(x, pd.Series):
        return x.apply(check_email)
    else:
        return check_email(x)

def check_email(val: Union[str, float, Any]) -> bool:
    """
    Function to check whether a value is a valid email
    """
    # pylint: disable=too-many-return-statements, too-many-branches
    if val in NULL_VALUES:
        return False

    if '@' not in val:
        return False

    user_part, domain_part = val.rsplit('@', 1)

    if not user_regex.match(user_part):
        return False

    if len(user_part.encode("utf-8")) > 64:
        return False

    if domain_part not in domain_whitelist and not domain_regex.match(domain_part):
        # Try for possible IDN domain-part
        try:
            domain_part = domain_part.encode('idna').decode('ascii')
            return domain_regex.match(domain_part) != None
        except UnicodeError:
            return False
    return True

def not_email(row: pd.Series, col: str, split: bool, errtype: str, processtype: str) -> pd.Series:
    """
    Return result when value unable to be parsed
    """
    
    if processtype == 'coerce':
        STATS[errtype] += 1
        if split:
            row["username"], row["domain"] = "None", "None"
        else:
            row[f"{col}_clean"] = "None"
    if processtype == 'ignore':
        STATS[errtype] += 1
        if split:
            row["username"], row["domain"] = row[col], "None"
        else:
            row[f"{col}_clean"] = row[col]
    return row

def reset_stats() -> None:
    """
    Reset global statistics dictionary
    """
    STATS["cleaned"] = 0
    STATS["null"] = 0
    STATS["unknown"] = 0
    

def report(nrows: int) -> None:
    """
    Describe what was done in the cleaning process
    """
    print("Email Cleaning Report:")
    if STATS["cleaned"] > 0:
        nclnd = STATS["cleaned"]
        pclnd = round(nclnd / nrows * 100, 2)
        print(f"\t{nclnd} values cleaned ({pclnd}%)")
    if STATS["unknown"] > 0:
        nunknown = STATS["unknown"]
        punknown = round(nunknown / nrows * 100, 2)
        print(f"\t{nunknown} values unable to be parsed ({punknown}%)")
    nnull = STATS["null"] + STATS["unknown"]
    pnull = round(nnull / nrows * 100, 2)
    ncorrect = nrows - nnull
    pcorrect = round(100 - pnull, 2)
    print(
        f"""Result contains {ncorrect} ({pcorrect}%) values in the correct format and {nnull} null values ({pnull}%)"""
    )

