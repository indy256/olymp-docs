// $Id: jstools.js,v 1.26 2008/04/03 17:59:37 nedjo Exp $

Drupal.preventSelect = function (elt) {
  // IE hack to prevent selection of the text when users click.
  if (document.onselectstart) {
    elt.onselectstart = function () {
      return false;
    }
  }
  else {
    $(elt).mousedown(function () {
      return false;
    });
  }
};

Drupal.url = function (path, query, fragment) {
  query = query ? query : '';
  fragment = fragment ? '#' + fragment : '';
  var base = Drupal.settings.basePath;
  if (!Drupal.settings.jstools.cleanurls) {
    if (query) {
      return base + '?q=' + path + '&' + query + fragment;
    }
    else {
      return base + '?q=' + path + fragment;
    }
  }
  else {
    if (query) {
      return base + path + '?' + query + fragment;
    }
    else {
      return base + path + fragment;
    }
  }
};

/**
 * Return the Drupal path portion of an href.
 */
Drupal.getPath = function (href) {
  href = Drupal.pathPortion(href);
  // 3 is the length of the '?q=' added to the url without clean urls.
  href = href.substring(Drupal.settings.basePath.length + (Drupal.settings.jstools.cleanurls ? 0 : 3), href.length);
  var chars = ['#', '?', '&'];
  for (i in chars) {
    if (href.indexOf(chars[i]) > -1) {
      href = href.substr(0, href.indexOf(chars[i]));
    }
  }
  return href;
};

/**
 * Add a segment to the beginning of a path.
 */
Drupal.prependPath = function (href, segment) {
  href = Drupal.pathPortion(href);
  // 3 is the length of the '?q=' added to the url without clean urls.
  var baseLength = Drupal.settings.jstools.basePath.length + (Drupal.settings.jstools.cleanurls ? 0 : 3);
  var base = href.substring(0, baseLength);
  return base + segment + '/' + href.substring(baseLength, href.length);
};

/**
 * Strip off the protocol plus domain from an href.
 */
Drupal.pathPortion = function (href) {
  // Remove e.g. http://example.com if present.
  var protocol = window.location.protocol;
  if (href.substring(0, protocol.length) == protocol) {
    // 2 is the length of the '//' that normally follows the protocol
    href = href.substring(href.indexOf('/', protocol.length + 2));
  }
  return href;
};

/**
 * Scroll to a given element's vertical page position.
 */
Drupal.scrollTo = function(el) {
  var pos = Drupal.absolutePosition(el);
  window.scrollTo(0, pos.y);
};

Drupal.elementChildren = function (element) {
  var children = [];
  for (i in element) {
    if (i.substr(0, 1) != '#') {
      children[children.length] = i;
    }
  }
  return children;
};

Drupal.elementProperties = function (element) {
  var properties = [];
  for (i in element) {
    if (i.substr(0, 1) == '#') {
      properties[properties.length] = i;
    }
  }
  return properties;
};

Drupal.parseQueryString = function (href) {
  query = Drupal.getQueryString(href);
  var args = {};
  var pairs = query.split("&");
  for(var i = 0; i < pairs.length; i++) {
    var pos = pairs[i].indexOf('=');
    if (pos == -1) continue;
    var argname = pairs[i].substring(0, pos);
    var value = pairs[i].substring(pos + 1);
    args[argname] = unescape(value.replace(/\+/g, " "));
  }
  return args;
};

Drupal.getQueryString = function (href) {
  if (href) {
    var index = href.indexOf('?');
    href = (index == -1) ? '' : href.substring(index + 1);
  }
  query = href ? href : location.search.substring(1);
  if (!Drupal.settings.jstools.cleanurls) {
    var index = query.indexOf('&');
    query = (index == -1) ? '' : query.substring(index + 1);
  }
  return query;
};

Drupal.pathMatch = function (path, paths, type) {
  // Convert paths into a regular expression.
  paths = '^' + paths + '$';
  paths = paths.replace(/\n/g, '$|^');
  paths = paths.replace(/\*/g, '.*');
  var search = path.search(new RegExp(paths)) > -1 ? true : false;
  return (type == 0) ? search : !search;
};

/**
 * Retrieves the absolute position of an element on the screen
 */
Drupal.absolutePosition = function (el) {
  var sLeft = 0, sTop = 0;
  var isDiv = /^div$/i.test(el.tagName);
  if (isDiv && el.scrollLeft) {
    sLeft = el.scrollLeft;
  }
  if (isDiv && el.scrollTop) {
    sTop = el.scrollTop;
  }
  var r = { x: el.offsetLeft - sLeft, y: el.offsetTop - sTop };
  if (el.offsetParent) {
    var tmp = Drupal.absolutePosition(el.offsetParent);
    r.x += tmp.x;
    r.y += tmp.y;
  }
  return r;
};

/**
 *  Returns the position of the mouse cursor based on the event object passed
 */
Drupal.mousePosition = function(e) {
  return { x: e.clientX + document.documentElement.scrollLeft, y: e.clientY + document.documentElement.scrollTop };
};