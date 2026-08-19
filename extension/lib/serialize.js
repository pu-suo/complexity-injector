// A one-at-a-time queue for work that cannot overlap.
//
// ONNX Runtime Web rejects a second run() while one is in flight, with
// "Session already started". Every tab the user has open talks to the same
// offscreen document and the same session, so concurrent requests are the
// normal case rather than an edge case.

export function makeQueue() {
  let chain = Promise.resolve();
  return function run(work) {
    // Chain on both settle paths so one rejection cannot stall the queue,
    // and swallow the tail separately so a caller's failure is still theirs
    // to handle rather than becoming an unhandled rejection here.
    const next = chain.then(work, work);
    chain = next.then(() => {}, () => {});
    return next;
  };
}
