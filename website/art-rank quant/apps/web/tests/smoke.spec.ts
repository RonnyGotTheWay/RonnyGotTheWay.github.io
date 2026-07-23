import {expect, test} from "@playwright/test";
import {mockThreePageApi} from "./candidate-api";

test("brand entrance stays isolated until the user enters the research system", async ({page}) => {
  const apiRequests: string[] = [];
  page.on("request", request => {
    if (request.url().includes("/api/v1/")) apiRequests.push(request.url());
  });
  await mockThreePageApi(page);
  await page.goto("/");
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByRole("heading", {name: "从市场噪声中， 识别下一步秩序"})).toBeVisible();
  await expect(page.getByRole("navigation")).toHaveCount(0);
  expect(apiRequests).toHaveLength(0);

  const enter = page.getByRole("link", {name: "进入 ART‑RANK"});
  await expect(enter).toBeVisible();
  await enter.click();
  await expect(page).toHaveURL(/\/weekly-market$/);

  const nav = page.locator("nav");
  await expect(nav.locator(":scope > a")).toHaveCount(2);
  await expect(nav.getByText("市场两月报告")).toBeVisible();
  await expect(nav.getByText("Top 10")).toBeVisible();
  await expect(nav.getByText("高级工具")).toHaveCount(0);
  await expect(nav.getByRole("link", {name: "市场两月报告"})).toHaveAttribute("aria-current", "page");
});

test("dashboard uses devil cursors while the landing page and text input do not", async ({page}) => {
  await mockThreePageApi(page);
  await page.goto("/");
  const landingCursor = await page.locator(".landing-page").evaluate(element => getComputedStyle(element).cursor);
  expect(landingCursor).not.toContain("devil-default.svg");

  await page.goto("/weekly-market");
  const shellCursor = await page.locator(".devil-cursor-zone").evaluate(element => getComputedStyle(element).cursor);
  const linkCursor = await page.getByRole("link", {name: "Top 10"}).evaluate(element => getComputedStyle(element).cursor);
  expect(shellCursor).toContain("devil-default.svg");
  expect(linkCursor).toContain("devil-active.svg");
});

test("mobile dashboard keeps both weekly destinations visible", async ({page}) => {
  await page.setViewportSize({width: 390, height: 844});
  await mockThreePageApi(page);
  await page.goto("/weekly-market");
  const nav = page.getByRole("navigation", {name: "研究系统导航"});
  await expect(nav).toBeVisible();
  await expect(nav.locator(":scope > a")).toHaveCount(2);
});
