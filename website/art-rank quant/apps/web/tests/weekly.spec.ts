import {expect, test} from "@playwright/test";

import {mockThreePageApi} from "./candidate-api";

test("weekly market renders four static charts without realtime requests", async ({page}) => {
  const realtimeRequests: string[] = [];
  page.on("request", request => {
    if (request.url().includes("/market/refresh") || request.url().includes("/models/bootstrap")) {
      realtimeRequests.push(request.url());
    }
  });
  await mockThreePageApi(page);
  await page.goto("/weekly-market");

  await expect(page.getByRole("heading", {name: "市场两月报告"})).toBeVisible();
  await expect(page.getByText("2026-04-23 — 2026-07-15")).toBeVisible();
  await expect(page.getByRole("heading", {name: "市场宽度"})).toBeVisible();
  await expect(page.getByRole("heading", {name: "全市场成交额"})).toBeVisible();
  await expect(page.getByRole("heading", {name: "个股60日收益分布"})).toBeVisible();
  await expect(page.getByRole("heading", {name: "行业60日涨跌热力图"})).toBeVisible();
  await expect(page.getByRole("heading", {name: "板块整体趋势"})).toBeVisible();
  await expect(page.locator(".weekly-chart-grid section")).toHaveCount(4);
  await expect(page.locator("canvas").first()).toBeVisible();
  expect(realtimeRequests).toHaveLength(0);
});

test("weekly ranking displays exactly ten model selections and version", async ({page}) => {
  await mockThreePageApi(page);
  await page.goto("/weekly-ranking");

  await expect(page.getByRole("heading", {name: "Top 10", exact: true})).toBeVisible();
  await expect(page.getByText("art-rank-weekly-test").first()).toBeVisible();
  await expect(page.getByText("ART 65% · 策略 35%")).toBeVisible();
  await expect(page.getByRole("columnheader", {name: "总评分"})).toBeVisible();
  await expect(page.getByRole("columnheader", {name: "研究参考买入"})).toBeVisible();
  await expect(page.getByRole("columnheader", {name: "目标价格"})).toBeVisible();
  await expect(page.getByText("9.80–10.20").first()).toBeVisible();
  await expect(page.getByText("19.70–20.20（参考值）").first()).toBeVisible();
  await expect(page.getByText("6.00–10.90%（参考值）").first()).toBeVisible();
  await expect(page.locator(".card.section tbody tr")).toHaveCount(10);
  await expect(page.locator(".card.section").getByText("测试股票1", {exact: true})).toBeVisible();
  await expect(page.locator(".card.section").getByText("测试股票10", {exact: true})).toBeVisible();
  await expect(page.getByRole("heading", {name: "板块 Top 10"})).toBeVisible();
});
